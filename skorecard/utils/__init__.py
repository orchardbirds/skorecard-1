from .arrayfuncs import assure_numpy_array
from .arrayfuncs import reshape_1d_to_2d
from .exceptions import DimensionalityError, UnknownCategoryError

__all__ = [
    "assure_numpy_array",
    "reshape_1d_to_2d",
    "DimensionalityError",
    "UnknownCategoryError",
    "NotInstalledError",
]
