from skorecard import datasets
from skorecard.preprocessing import TreeBucketTransformer

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
    """Test that the correct number of bucketers are created."""
    X = df[["LIMIT_BAL", "BILL_AMT1"]].values
    y = df["default"]
    tbt = TreeBucketTransformer(
        inf_edges=True,
        max_depth=4,
        criterion="entropy",
        min_samples_leaf=20,  # Minimum number of entries in the bins
        min_impurity_decrease=0.001,
    )
    tbt.fit(X, y)

    assert len(tbt.BucketDict) == 2
    assert tbt.BucketDict["Feature_0"].bin_count > 0
    assert tbt.BucketDict["Feature_1"].bin_count > 0

    return None


def test_transform(df):
    """Test that the correct shape is returned."""
    X = df[["LIMIT_BAL", "BILL_AMT1"]].values
    y = df["default"]
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
