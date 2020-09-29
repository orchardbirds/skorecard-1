# import numpy as np
# from sklearn.compose import ColumnTransformer

import pytest
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.exceptions import NotFittedError

from skorecard import datasets
from skorecard.bucketers import EqualWidthBucketer, EqualFrequencyBucketer


@pytest.fixture()
def df():
    """Generate dataframe."""
    return datasets.load_uci_credit_card(as_frame=True)


# TODO: write tests with different kinds of sklearn pipelines
# - nested
# - ColumnTransformer and ColumnSelector usage


def test_make_pipeline(df):
    """Make sure bucketers work inside a pipeline."""
    y = df["default"].values
    X = df.drop(columns=["default"])

    pipe = make_pipeline(
        EqualWidthBucketer(bins=4, variables=["LIMIT_BAL"]), EqualFrequencyBucketer(bins=7, variables=["BILL_AMT1"]),
    )
    new_X = pipe.fit_transform(X, y)
    assert isinstance(new_X, pd.DataFrame)


def test_pipeline_errors(df):
    """Make sure incorrect input also throws correct errors in pipeline."""
    y = df["default"].values
    X = df.drop(columns=["default"])

    bu = EqualWidthBucketer(bins=4, variables=["LIMIT_BAL", "BILL_AMT1"])
    with pytest.raises(NotFittedError):
        bu.transform(X)  # not fitted yet
    with pytest.raises(TypeError):
        bu.fit_transform(np.array([1, 2, 3]), y)


# def test_bucket_transformer_bin_count_list(df):
#     """Test the exception is raised in scikit-learn pipeline."""
#     with pytest.raises(AttributeError):
#         transformer = ColumnTransformer(
#             transformers=[
#                 ("simple", SimpleBucketTransformer(bin_count=2), [1]),
#                 ("agglom", AgglomerativeBucketTransformer(bin_count=4), [0]),
#                 ("quantile", QuantileBucketTransformer(bin_count=[10]), [3]),
#             ],
#             remainder="passthrough",
#         )
#         transformer.fit_transform(df.values)

#     return None


# def test_bucket_transformer_exception(df):
#     """Test the exception is raised in scikit-learn pipeline."""
#     with pytest.raises(DimensionalityError):
#         transformer = ColumnTransformer(
#             transformers=[
#                 ("simple", SimpleBucketTransformer(bin_count=2), [1]),
#                 ("agglom", AgglomerativeBucketTransformer(bin_count=4), [0]),
#                 ("quantile", QuantileBucketTransformer(bin_count=10), [2, 3]),
#             ],
#             remainder="passthrough",
#         )
#         transformer.fit_transform(df.values)

#     return None


# def test_bucket_transformer(df):
#     """Test that we can utilise the main bucket transformers in a scikit-learn pipeline."""
#     transformer = ColumnTransformer(
#         transformers=[
#             ("simple", SimpleBucketTransformer(bin_count=2), [1]),
#             ("agglom", AgglomerativeBucketTransformer(bin_count=4), [0]),
#             ("quantile_0", QuantileBucketTransformer(bin_count=10), [2]),
#             ("quantile_1", QuantileBucketTransformer(bin_count=6), [3]),
#         ],
#         remainder="passthrough",
#     )

#     X = transformer.fit_transform(df.values)

#     # Test only non-categorical variables
#     assert len(np.unique(X[:, 2])) == 10
#     assert len(np.unique(X[:, 3])) == 6
#     assert np.all(X[:, 4] == df["default"].values)

#     return None
