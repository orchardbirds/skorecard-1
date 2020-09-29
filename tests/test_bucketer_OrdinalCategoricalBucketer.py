from skorecard import datasets
from skorecard.bucketers import OrdinalCategoricalBucketer

# from sklearn.pipeline import Pipeline, FeatureUnion
# from skorecard.pipeline import ColumnSelector
import numpy as np
import pytest


@pytest.fixture()
def df():
    """Generate dataframe."""
    return datasets.load_uci_credit_card(as_frame=True)


def test_threshold_min(df) -> None:
    """Test that threshold_min < 1 raises an error."""
    with pytest.raises(ValueError):
        OrdinalCategoricalBucketer(tol=-0.1, variables=["EDUCATION"])
    with pytest.raises(ValueError):
        OrdinalCategoricalBucketer(tol=1.001, variables=["EDUCATION"])


def test_correct_output(df):
    """Test that correct use of CatBucketTransformer returns expected results."""
    X = df
    y = df["default"].values

    cbt = OrdinalCategoricalBucketer(tol=0.44, variables=["EDUCATION"])
    cbt.fit(X, y)
    X_trans = cbt.transform(X)
    assert len(X_trans["EDUCATION"].unique()) == 2

    cbt = OrdinalCategoricalBucketer(tol=0.05, variables=["EDUCATION"])
    cbt.fit(X, y)
    X_trans = cbt.transform(X)
    assert len(X_trans["EDUCATION"].unique()) == 4

    cbt = OrdinalCategoricalBucketer(tol=0, variables=["EDUCATION"])
    cbt.fit(X, y)
    X_trans = cbt.transform(X)
    assert len(X_trans["EDUCATION"].unique()) == len(X["EDUCATION"].unique())

    # when the threshold is above the maximum value, make sure its only one bucket
    cbt = OrdinalCategoricalBucketer(tol=0.5, variables=["EDUCATION"])
    cbt.fit(X, y)
    X_trans = cbt.transform(X)
    assert X_trans["EDUCATION"].unique() == np.array([0])


def test_mapping_dict(df):
    """Test that the mapping dict is created correctly."""
    X = df
    y = df["default"].values
    cbt = OrdinalCategoricalBucketer(tol=0, variables=["EDUCATION"])
    cbt.fit(X, y)
    bucket_map = cbt.features_bucket_mapping_.get("EDUCATION")
    assert len(bucket_map.map) == len(np.unique(X["EDUCATION"]))
