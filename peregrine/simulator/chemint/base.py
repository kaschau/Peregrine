from peregrine.config import ConfigError

templates = f"{__package__}.kernels"


class BaseChemistry:
    name = None

    def __init__(self, cfgsect, mixture):
        self.cfgsect, self.mixture = cfgsect, mixture

    def validate(self):
        pass

    def arrays(self):
        return {}

    def structs(self):
        return {}

    def tplargs(self):
        return {}

    # the kernels are made in begin_dQ_graph, per register pair; only the provider is kept here
    def decl_kerns(self, provider):
        self.provider = provider

    ###########################################################################
    # Slot graphs
    ###########################################################################
    def begin_dQ_graph(self, g, Q, dQ):
        pass


class NoChemistry(BaseChemistry):
    pass


class BaseReactingChemistry(BaseChemistry):
    def validate(self):
        if self.mixture.chemistry is None:
            raise ConfigError("chem-integration.kind", f"{self.name} needs mixture.chemistry")

    def tplargs(self):
        mixture = self.mixture
        return {"rxns": mixture.chemistry.terms(mixture.species), "chemistry": mixture.chemistry.template()}
