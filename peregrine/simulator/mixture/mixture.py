from pathlib import Path

import numpy as np

from peregrine.simulator.mixture.parser import get_parser
from peregrine.codegen import Struct
from peregrine.misc import merge_disjoint, subclass_where
from peregrine.simulator.mixture.constants import Ru
from peregrine.simulator.mixture.base import BaseModel
from peregrine.simulator.mixture.chemistry import BaseChemistryModel
from peregrine.simulator.mixture.diffusion import BaseSpeciesDiffusionModel
from peregrine.simulator.mixture.eos import BaseEosModel
from peregrine.simulator.mixture.mixrule import BaseMixingRuleModel
from peregrine.simulator.mixture.species import Species
from peregrine.simulator.mixture.transport import BaseTransportModel


class Mixture:
    def __init__(self, cfgsect, root=None):
        self.cfgsect = cfgsect
        eos = subclass_where(BaseEosModel, name=cfgsect["eos"])
        self.eos = eos(cfgsect)
        trans, diffusion = cfgsect["trans"], cfgsect["diffusion"]
        self.trans = self.diffusion = self.mixrule = None
        if trans:
            kind = subclass_where(BaseTransportModel, name=trans)
            self.trans = kind(cfgsect)
            kind = subclass_where(BaseSpeciesDiffusionModel, name=diffusion)
            self.diffusion = kind(cfgsect)
            mixrule = cfgsect["mixing-rule"]
            kind = subclass_where(BaseMixingRuleModel, name=mixrule)
            self.mixrule = kind(cfgsect)

        usersp, self.reactions = self.read(cfgsect, root)
        chemistry = cfgsect["chemistry"]
        self.chemistry = subclass_where(BaseChemistryModel, name=chemistry)(cfgsect, self.reactions) if chemistry else None
        self._check_deps()
        self.species = Species.build(usersp, self.models)
        self.spnames = list(self.species)
        self.ns = len(self.spnames)

        for model in self.models:
            model.check(self.species)

        # each model stores what it works out on the species, eos first
        for model in self.models:
            model.populate(self.species)

    # the last species is what the others leave, so the solution never carries it
    @property
    def carriedSpecies(self):
        return self.spnames[:-1]

    def constants(self):
        MW = BaseModel.collect("MW", self.species)
        own = dict(Ru=np.array(Ru), MW=MW, MWinv=1.0 / MW, Rsp=Ru / MW)
        theirs = (m.constants(self.species) for m in self.models)
        return merge_disjoint(own, *theirs)

    def structs(self):
        constants = self.constants()
        summed = {m: constants[c] for m, c in self.eos.moments.items()}
        state = Struct("EosState", self.eos.computes(self.ns))
        fields = [m for m, a in summed.items() if a.ndim == 1]
        arrays = {m: a.shape[1] for m, a in summed.items() if a.ndim == 2}
        moments = Struct("MixMom", fields, arrays)
        own = {"EosState": state, "MixMom": moments}
        theirs = ({s.name: s for s in m.structs(self.species)} for m in self.models)
        return merge_disjoint(own, *theirs)

    # the order they are checked and populated in: the eos first, the others read what it stores
    @property
    def models(self):
        models = (self.eos, self.trans, self.diffusion, self.mixrule, self.chemistry)
        return tuple(m for m in models if m is not None)

    def _check_deps(self):
        parts = [*self.models, self.eos.caloric]
        for model in self.models:
            for need in model.deps:
                if not any(isinstance(part, need) for part in parts):
                    have = ", ".join(type(p).__name__ for p in parts)
                    lack = f"{model.name} needs {need.__name__}"
                    raise ValueError(f"{lack}; the case has {have}")

    def __repr__(self):
        return f"<{type(self).__name__} {', '.join(self.described())}>"

    def described(self):
        models = [self.eos.name]
        if self.trans is not None:
            names = (self.trans.name, self.diffusion.name, self.mixrule.name)
            models.append("/".join(names))
        if self.chemistry is not None:
            models.append(f"{self.chemistry.name} ({len(self.reactions)} reactions)")
        return [f"{self.ns} species", " + ".join(models)]

    @staticmethod
    def read(cfgsect, root):
        mixture = cfgsect["species"]
        if isinstance(mixture, dict):
            return dict(mixture), []
        if isinstance(mixture, (list, tuple)):
            return {sp: {} for sp in mixture}, []

        here = Path(__file__).parent
        locs = [] if root is None else [Path(root) / mixture]
        locs += [Path(mixture), here / "database" / "mechanisms" / mixture]
        for loc in locs:
            if loc.is_file():
                parsed = get_parser(loc)(str(loc))
                return parsed.species, parsed.reactions
        tried = ", ".join(str(p) for p in locs)
        msg = f"Cannot find the mixture {mixture!r}. Tried {tried}"
        raise FileNotFoundError(msg)
