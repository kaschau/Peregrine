from peregrine.misc import subclass_where
from peregrine.backend import abi, cuda, hip, openmp  # noqa: F401
from peregrine.backend.allocation import Allocation, Scalar, Subview
from peregrine.backend.base.backend import BaseBackend
from peregrine.backend.base.compiler import BaseCompiler


def get_backend(name, cfg):
    return subclass_where(BaseBackend, name=name)(cfg)


__all__ = [
    "abi",
    "Allocation",
    "BaseBackend",
    "BaseCompiler",
    "Scalar",
    "Subview",
    "get_backend",
]
