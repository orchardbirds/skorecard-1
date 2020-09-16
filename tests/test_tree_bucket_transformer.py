from skorecard import datasets
from skorecard.preprocessing import TreeBucketTransformer
from skorecard.utils.exceptions import DimensionalityError

import numpy as np
import pytest


@pytest.fixture()
def df():
    """Generate dataframe."""
    return datasets.load_uci_credit_card(as_frame=True)


def test_kwargs_are_saved():
    """Test that the kwargs fed to the TreeBucketTransformer are saved."""
    tbt = TreeBucketTransformer(
        inf_edges=True, max_depth=4, criterion="entropy", min_samples_leaf=20, min_impurity_decrease=0.001
    )

    assert tbt.kwargs["inf_edges"]
    assert tbt.kwargs["max_depth"] == 4
    assert tbt.kwargs["criterion"] == "entropy"
    assert tbt.kwargs["min_samples_leaf"] == 20
    assert tbt.kwargs["min_impurity_decrease"] == 0.001

    return None


def test_BucketDict(df):
    """Test that the exception is raised with mutiple bins of bucketers are created."""
    X = df[["LIMIT_BAL", "BILL_AMT1"]].values
    y = df["default"]
    tbt = TreeBucketTransformer(
        inf_edges=True,
        max_depth=4,
        criterion="entropy",
        min_samples_leaf=20,  # Minimum number of entries in the bins
        min_impurity_decrease=0.001,
    )
    with pytest.raises(DimensionalityError):
        tbt.fit(X, y)


def test_transform(df):
    """Test that the correct shape is returned."""
    X = df["LIMIT_BAL"].values
    y = df["default"].values
    tbt = TreeBucketTransformer(
        inf_edges=True,
        max_depth=4,
        criterion="entropy",
        min_samples_leaf=20,  # Minimum number of entries in the bins
        min_impurity_decrease=0.001,
    )
    tbt.fit(X, y)

    assert tbt.transform(X).shape == X.shape

    return None


def test_dimensionality_exception(df):
    """Test the exception is raised if too many features are run at the same time."""
    tbt = TreeBucketTransformer()
    with pytest.raises(DimensionalityError):
        tbt.fit_transform(df[["LIMIT_BAL", "MARRIAGE"]].values)
    return None


def test_inifinte_edges(df):
    """Test that infinite edges actually are infinite."""
    SBT = TreeBucketTransformer(
        max_depth=4,
        criterion="entropy",
        min_samples_leaf=20,  # Minimum number of entries in the bins
        min_impurity_decrease=0.001,
        infinite_edges=True,
    )
    SBT.fit(df["LIMIT_BAL"].values, df["default"].values)
    assert SBT.boundaries[0] == -np.inf
    assert SBT.boundaries[-1] == np.inf

    SBT = TreeBucketTransformer(
        max_depth=4,
        criterion="entropy",
        min_samples_leaf=20,  # Minimum number of entries in the bins
        min_impurity_decrease=0.001,
        infinite_edges=False,
    )
    SBT.fit(df["LIMIT_BAL"].values, df["default"].values)
    assert SBT.boundaries[0] != -np.inf
    assert SBT.boundaries[-1] != np.inf
