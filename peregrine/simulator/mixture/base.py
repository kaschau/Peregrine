import numpy as np

from peregrine.simulator.mixture.species import AnyOf


class BaseModel:
    # what the config calls it, on the class so subclass_where can find it
    name = None
    # the structs its macros carry a point in
    def structs(self, species):
        return ()

    @classmethod
    def template(cls):
        package, module = cls.__module__.rsplit(".", 1)
        return f"{package}.kernels.{module}"

    def __init__(self, cfgsect):
        # the case's mixture section: what every fit is made over and to
        self.cfgsect = cfgsect
        # what it reads off each species; everything needs MW
        self.needs = ("MW",)
        # what the case must state per species: modelling choices the library never carries
        self.fromcase = ()
        # which of needs may be absent if what derives them is present
        self.derivable = {}
        # {method: the quantity, or quantities, it returns}, stored on the species in this order
        self.provides = {}
        # the model classes this one only works alongside
        self.deps = ()

    def check(self, species):
        soft = [key for keys in self.derivable for key in keys]
        hard = [key for key in self.needs if key not in soft]
        missing = self._missing(hard, species)
        if missing:
            raise ValueError(f"{self.name} needs {self._shortfall(missing)}")
        for derived, inputs in self.derivable.items():
            # a species with the set does not need the inputs
            lacking = self._missing(derived, species)
            short = [n for n in lacking if species[n].missing(inputs)]
            if short:
                raise ValueError(
                    f"{self.name} needs {', '.join(derived)}, "
                    f"or {', '.join(inputs)} to derive them, "
                    f"for {self._listed(short)}"
                )
        absent = self._missing(self.fromcase, species)
        if absent:
            raise ValueError(f"{self.name} needs {self._shortfall(absent)}")

    def populate(self, species):
        for method, quantities in self.provides.items():
            returned = getattr(self, method)(species)
            if isinstance(quantities, str):
                quantities, returned = (quantities,), (returned,)
            for quantity, values in zip(quantities, returned):
                for sp, value in zip(species.values(), values):
                    sp[quantity] = value

    def constants(self, species):
        return {}

    # the models fit over these and nothing clamps T to them: a fit evaluated outside extrapolates silently
    @property
    def trange(self):
        return tuple(self.cfgsect["T-range"])

    @staticmethod
    def collect(key, species):
        prop = [sp[key] for sp in species.values()]
        return prop if isinstance(prop[0], str) else np.array(prop, float)

    @classmethod
    def collect_poly(cls, key, species):
        return cls.padded([sp[key] for sp in species.values()])

    # zero past each row's last term, so a kernel walks every row with one trip count
    @staticmethod
    def padded(rows, dtype=np.float64):
        terms = max((len(r) for r in rows), default=1)
        array = np.zeros((len(rows), max(terms, 1)), dtype=dtype)
        for k, r in enumerate(rows):
            array[k, : len(r)] = r
        return array

    @staticmethod
    def _missing(keys, species):
        missing = {}
        for name, sp in species.items():
            absent = sp.missing(keys)
            if absent:
                missing[name] = absent
        return missing

    def _shortfall(self, missing):
        grouped = {}
        for name, keys in missing.items():
            grouped.setdefault(tuple(keys), []).append(name)
        return "; ".join(
            f"{', '.join('|'.join(k) if isinstance(k, AnyOf) else k for k in keys)} "
            f"for {self._listed(names)}"
            for keys, names in grouped.items()
        )

    @staticmethod
    def _listed(names, most=6):
        if len(names) <= most:
            return ", ".join(names)
        return ", ".join(names[:most]) + f" and {len(names) - most} more"
