import yaml

from peregrine.simulator.mixture.parser.base import BaseParser


# YAML 1.1 reads NO/YES/ON/OFF as booleans, which breaks the species 'NO'
class _CanteraLoader(yaml.SafeLoader):
    pass


_CanteraLoader.yaml_implicit_resolvers = {
    k: [(tag, regexp) for tag, regexp in v if tag != "tag:yaml.org,2002:bool"]
    for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


class CanteraParser(BaseParser):
    name = "cantera"
    suffixes = (".yaml", ".yml")
    # a yaml with no units section is in SI
    defunits = {
        "length": "m", "quantity": "kmol", "activation-energy": "J/kmol"
    }

    def load(self, path):
        with open(path) as f:
            return yaml.load(f, Loader=_CanteraLoader)
