from .arrayfuncs import (
    reshape_1d_to_2d,
    convert_sparse_matrix,
)
from .exceptions import (
    DimensionalityError,
    UnknownCategoryError,
    NotInstalledError,
    NotPreBucketedError,
    NotBucketObjectError,
    BucketingPipelineError,
)
from .dataframe import detect_types

__all__ = [
    "reshape_1d_to_2d",
    "convert_sparse_matrix",
    "DimensionalityError",
    "UnknownCategoryError",
    "NotInstalledError",
    "NotBucketObjectError",
    "detect_types",
    "NotPreBucketedError",
    "BucketingPipelineError",
]
