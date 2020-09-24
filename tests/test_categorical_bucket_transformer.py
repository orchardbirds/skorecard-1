from skorecard import datasets
from skorecard.preprocessing import CatBucketTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from skorecard.pipeline import ColumnSelector
import numpy as np
import pytest


@pytest.fixture()
def df():
    """Generate dataframe."""
    return datasets.load_uci_credit_card(as_frame=True)


def test_threshold_min(df):
    """Test that threshold_min < 1 raises an error."""
    with pytest.raises(ValueError):
        CatBucketTransformer(threshold_min=-0.1)

    return None


def test_correct_output(df):
    """Test that correct use of CatBucketTransformer returns expected results."""
    X = df["EDUCATION"].values
    y = df["default"].values
    cbt = CatBucketTransformer(threshold_min=0.44)
    cbt.fit(X, y)
    X_trans = cbt.transform(X)
    assert len(np.unique(X_trans)) == 2

    cbt = CatBucketTransformer(threshold_min=0.05)
    cbt.fit(X, y)
    X_trans = cbt.transform(X)
    assert len(np.unique(X_trans)) == 4

    cbt = CatBucketTransformer(threshold_min=0.00)
    cbt.fit(X, y)
    X_trans = cbt.transform(X)

    assert len(np.unique(X)) == len(np.unique(X_trans))

    # when the thershold is above the maximum value, make sure its only one bucket
    cbt = CatBucketTransformer(threshold_min=0.5)
    cbt.fit(X, y)
    X_ = cbt.transform(X)
    assert np.unique(X_) == np.array([0])

    return None


def test_mapping_dict(df):
    """Test that the mapping dict is created correctly."""
    X = df["EDUCATION"].values
    y = df["default"].values
    cbt = CatBucketTransformer(threshold_min=0.05)
    cbt.fit(X, y)
    cbt.transform(X)

    assert len(cbt.map) == len(np.unique(X))

    return None


def test_correct_shape_on_2d_array(df):
    """Test CatBucketTransformer can be applied in a pipeline for 2D-arrays."""
    cat_cols = ["MARRIAGE", "EDUCATION"]

    X = df[cat_cols]

    cat_transformer_list = [
        (
            f"cat_bkt_{col}",
            Pipeline([("selector", ColumnSelector([col])), ("bucketer", CatBucketTransformer(threshold_min=0.01))]),
        )
        for col in cat_cols
    ]

    transformers = Pipeline([("bucketed_features", FeatureUnion(n_jobs=1, transformer_list=cat_transformer_list))])

    assert X.shape == transformers.fit_transform(X, None).shape
