# import numpy as np
# from sklearn.compose import ColumnTransformer

import pytest
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from skorecard import datasets
from skorecard.bucketers import (
    EqualWidthBucketer,
    EqualFrequencyBucketer,
    OrdinalCategoricalBucketer,
    DecisionTreeBucketer,
    OptimalBucketer,
)
from skorecard.pipeline import get_features_bucket_mapping, KeepPandas, make_coarse_classing_pipeline
from skorecard.bucket_mapping import BucketMapping


@pytest.fixture()
def df():
    """Generate dataframe."""
    return datasets.load_uci_credit_card(as_frame=True)


@pytest.mark.filterwarnings("ignore:sklearn.")
def test_keep_pandas(df, caplog):
    """Tests the KeepPandas() class."""
    y = df["default"].values
    X = df.drop(columns=["default"])

    bucket_pipeline = make_pipeline(StandardScaler(), EqualWidthBucketer(bins=5, variables=["LIMIT_BAL", "BILL_AMT1"]),)
    # Doesn't work, input should be a pandas dataframe.
    with pytest.raises(TypeError):
        bucket_pipeline.fit(X, y)

    bucket_pipeline = make_pipeline(
        KeepPandas(StandardScaler()), EqualWidthBucketer(bins=5, variables=["LIMIT_BAL", "BILL_AMT1"]),
    )

    with pytest.raises(NotFittedError):
        bucket_pipeline.transform(X)

    bucket_pipeline.fit(X, y)
    assert type(bucket_pipeline.transform(X)) == pd.DataFrame

    bucket_pipeline = ColumnTransformer(
        [
            ("categorical_preprocessing", OrdinalCategoricalBucketer(), ["EDUCATION", "MARRIAGE"]),
            ("numerical_preprocessing", EqualWidthBucketer(bins=5), ["LIMIT_BAL", "BILL_AMT1"]),
        ],
        remainder="passthrough",
    )

    # Make sure warning is raised
    caplog.clear()
    KeepPandas(make_pipeline(bucket_pipeline))
    assert "sklearn.compose.ColumnTransformer can change" in caplog.text

    # Make sure warning is raised
    caplog.clear()
    KeepPandas(bucket_pipeline)
    assert "sklearn.compose.ColumnTransformer can change" in caplog.text

    assert type(KeepPandas(bucket_pipeline).fit_transform(X)) == pd.DataFrame


def test_bucketing_pipeline(df):
    """Test the class."""
    y = df["default"].values
    X = df.drop(columns=["default"])

    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    prebucket_pipeline = make_pipeline(DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05))

    bucket_pipeline = make_coarse_classing_pipeline(
        OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
        OptimalBucketer(variables=cat_cols, max_n_bins=10, min_bin_size=0.05),
    )

    pipe = make_pipeline(prebucket_pipeline, bucket_pipeline)
    pipe.fit(X, y)
    # Make sure we can fit it twice
    pipe.fit(X, y)

    # make sure sure transforms work.
    pipe.transform(X)
    pipe.fit_transform(X, y)


def test_get_features_bucket_mapping(df):
    """Test retrieving info from sklearn pipeline."""
    y = df["default"].values
    X = df.drop(columns=["default"])

    nested_pipeline = make_pipeline(
        make_pipeline(EqualWidthBucketer(bins=5, variables=["LIMIT_BAL", "BILL_AMT1"])),
        OrdinalCategoricalBucketer(variables=["EDUCATION", "MARRIAGE"]),
    )

    with pytest.raises(NotFittedError):
        get_features_bucket_mapping(nested_pipeline)

    nested_pipeline.fit(X, y)
    bm = get_features_bucket_mapping(nested_pipeline)
    assert bm.get("EDUCATION") == BucketMapping(
        feature_name="EDUCATION", type="categorical", map={2: 1, 1: 2, 3: 3}, right=True
    )


# TODO: write tests with different kinds of sklearn pipelines
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
