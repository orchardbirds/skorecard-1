from skorecard.bucketers import OrdinalCategoricalBucketer

# from sklearn.pipeline import Pipeline, FeatureUnion
# from skorecard.pipeline import ColumnSelector
import numpy as np
import pytest


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
    assert X_trans["EDUCATION"].unique() == np.array([1])


def test_mapping_dict(df):
    """Test that the mapping dict is created correctly."""
    X = df
    y = df["default"].values
    cbt = OrdinalCategoricalBucketer(tol=0, variables=["EDUCATION"])
    cbt.fit(X, y)
    bucket_map = cbt.features_bucket_mapping_.get("EDUCATION")
    assert len(bucket_map.map) == len(np.unique(X["EDUCATION"]))


def test_encoding_method(df):
    """Test the encoding method."""
    X = df[["EDUCATION", "default"]]
    y = df["default"]

    ocb = OrdinalCategoricalBucketer(tol=0.03, variables=["EDUCATION"], encoding_method="frequency")
    ocb.fit(X, y)

    assert ocb.features_bucket_mapping_["EDUCATION"].map == {2: 0, 1: 1, 3: 2}

    ocb = OrdinalCategoricalBucketer(tol=0.03, variables=["EDUCATION"], encoding_method="ordered")
    ocb.fit(X, y)

    assert ocb.features_bucket_mapping_["EDUCATION"].map == {1: 0, 3: 1, 2: 2}


def test_specials(df):
    """Test specials get assigned to the highest bin."""
    X = df[["EDUCATION"]]
    y = df["default"]

    ocb = OrdinalCategoricalBucketer(
        tol=0.03, variables=["EDUCATION"], encoding_method="ordered", specials={"ed 0": [1]}
    )
    ocb.fit(X, y)

    X_transform = ocb.transform(X)
    assert np.unique(X_transform[X["EDUCATION"] == 1].values)[0] == 4

    ocb = OrdinalCategoricalBucketer(
        tol=0.03, variables=["EDUCATION"], encoding_method="frequency", specials={"ed 0": [1]}
    )
    ocb.fit(X, y)

    X_transform = ocb.transform(X)
    assert np.unique(X_transform[X["EDUCATION"] == 1].values)[0] == 4
