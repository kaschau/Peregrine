from pathlib import Path

from peregrine.misc import subclasses
from peregrine.simulator.mixture.parser.base import BaseParser
from peregrine.simulator.mixture.parser.cantera import CanteraParser
from peregrine.simulator.mixture.parser.chemkin import ChemkinParser

__all__ = ["BaseParser", "CanteraParser", "ChemkinParser", "get_parser"]


def get_parser(path):
    suffix = Path(path).suffix.lower()
    parsers = subclasses(BaseParser)
    return next(cls for cls in parsers if suffix in cls.suffixes)
