from skorecard import datasets
from skorecard.preprocessing import CatBucketTransformer
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


def test_threshold_max(df):
    """Test that threshold_max > 1 raises an error."""
    with pytest.raises(ValueError):
        CatBucketTransformer(threshold_max=1.1)

    return None


def test_bad_thresholds(df):
    """Test that threshold_min > threshold_max raises an error."""
    with pytest.raises(AttributeError):
        CatBucketTransformer(threshold_min=0.5, threshold_max=0.4)

    return None


def test_epsilon_min(df):
    """Test that epsilon < 1 raises an error."""
    with pytest.raises(ValueError):
        CatBucketTransformer(epsilon=-0.05)

    return None


def test_epsilon_max(df):
    """Test that epsilon > 1 raises an error."""
    with pytest.raises(ValueError):
        CatBucketTransformer(epsilon=1.05)

    return None


def test_correct_output(df):
    """Test that correct use of CatBucketTransformer returns expected results."""
    X = df["EDUCATION"].values
    y = df["default"].values
    cbt = CatBucketTransformer(threshold_min=0.01, threshold_max=0.2, epsilon=0.05)
    cbt.fit(X, y)
    X = cbt.transform(X)

    assert len(np.unique(X)) == 4

    X = df["EDUCATION"].values
    y = df["default"].values
    cbt = CatBucketTransformer(threshold_min=0.00, threshold_max=1.0, epsilon=0.00)
    cbt.fit(X, y)
    X = cbt.transform(X)

    assert len(np.unique(X)) == 1

    return None


def test_mapping_dict(df):
    """Test that the mapping dict is created correctly."""
    X = df["EDUCATION"].values
    y = df["default"].values
    cbt = CatBucketTransformer(threshold_min=0.01, threshold_max=0.2, epsilon=0.05)
    cbt.fit(X, y)
    cbt.transform(X)

    assert len(cbt.mapping_dict) == len(np.unique(X))

    return None
