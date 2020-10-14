from skorecard import datasets
from skorecard.bucketers import DecisionTreeBucketer

import pytest


@pytest.fixture()
def df():
    """Generate dataframe."""
    return datasets.load_uci_credit_card(as_frame=True)


def test_kwargs_are_saved():
    """Test that the kwargs fed to the TreeBucketTransformer are saved."""
    tbt = DecisionTreeBucketer(variables=["LIMIT_BAL"], criterion="entropy", min_impurity_decrease=0.001,)
    assert tbt.kwargs["criterion"] == "entropy"
    assert tbt.kwargs["min_impurity_decrease"] == 0.001


def test_transform(df):
    """Test that the correct shape is returned."""
    X = df[["LIMIT_BAL", "BILL_AMT1"]]
    y = df["default"]

    tbt = DecisionTreeBucketer(variables=["LIMIT_BAL", "BILL_AMT1"], criterion="entropy", min_impurity_decrease=0.001,)
    tbt.fit(X, y)

    assert tbt.transform(X).shape == X.shape