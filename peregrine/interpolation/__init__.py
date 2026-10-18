from peregrine.interpolation.base import BaseInterpolator
from peregrine.interpolation.nearest import NearestInterpolator
from peregrine.interpolation.rbf import RbfInterpolator
from peregrine.misc import subclass_where


def get_interpolator(kind="nearest", deep_search=False, **options):
    return subclass_where(BaseInterpolator, name=kind)(deep_search=deep_search, **options)


__all__ = [
    "BaseInterpolator",
    "NearestInterpolator",
    "RbfInterpolator",
    "get_interpolator",
]
