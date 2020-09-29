import numpy as np
import pytest
import pandas as pd

from skorecard import datasets
from skorecard.bucketers import WoEBucketer


@pytest.fixture()
def X_y():
    """Set of X,y for testing the transformers."""
    X = pd.DataFrame(
        np.array([[0, 1], [1, 0], [0, 0], [3, 2], [0, 1], [1, 2], [2, 0], [2, 1], [0, 0]], np.int32,),
        columns=["col1", "col2"],
    )
    y = pd.Series(np.array([0, 0, 0, 1, 1, 1, 0, 0, 1]))

    return X, y


@pytest.fixture()
def X_y_2():
    """Set of X,y for testing the transformers.

    In the first column, bucket 3 is not present in class 1.
    """
    X = pd.DataFrame(
        np.array([[0, 1], [1, 0], [0, 0], [3, 2], [0, 1], [1, 2], [2, 0], [2, 1], [0, 0]], np.int32,),
        columns=["col1", "col2"],
    )
    y = pd.Series(np.array([0, 0, 0, 0, 0, 1, 1, 1, 1]))

    return X, y


@pytest.fixture()
def df():
    """Generate dataframe."""
    return datasets.load_uci_credit_card(as_frame=True)


def test_woe_transformed_dimensions(X_y):
    """Tests that the total number of unique WOEs matches the unique number of bins in X."""
    X, y = X_y

    woeb = WoEBucketer(variables=["col1", "col2"])
    new_X = woeb.fit_transform(X, y)
    assert len(new_X["col1"].unique()) == len(X["col1"].unique())
    assert len(new_X["col2"].unique()) == len(X["col2"].unique())


def test_missing_bucket(X_y_2):
    """Tests that the total number of unique WOEs matches the unique number of bins in X."""
    X, y = X_y_2

    woeb = WoEBucketer(variables=["col1", "col2"])
    new_X = woeb.fit_transform(X, y)

    assert new_X.shape == X.shape
    assert len(new_X["col1"].unique()) == len(X["col1"].unique())
    assert len(new_X["col2"].unique()) == len(X["col2"].unique())

    # because class 1 will have zero counts, the WOE transformer will divide by the value of epsilon, avoinding infinite
    # numbers
    assert not any(new_X["col1"] == np.inf)
    assert not any(new_X["col2"] == np.inf)

    # If epsilon is set to zero, expect a Division By Zero exception
    with pytest.raises(ZeroDivisionError):
        WoEBucketer(epsilon=0, variables=["col1", "col2"]).fit_transform(X, y)


def test_woe_values(X_y):
    """Tests the value of the WOE."""
    X, y = X_y

    woeb = WoEBucketer(variables=["col1", "col2"])
    new_X = woeb.fit_transform(X, y)
    new_X

    expected = pd.DataFrame(
        {
            "col1": {
                0: 0.22309356256166854,
                1: 0.2230435962938856,
                2: 0.22309356256166854,
                3: 7.824445930877619,
                4: 0.22309356256166854,
                5: 0.2230435962938856,
                6: -8.294299608857235,
                7: -8.294299608857235,
                8: 0.22309356256166854,
            },
            "col2": {
                0: -0.46985367797961614,
                1: -0.8752354701118938,
                2: -0.8752354701118938,
                3: 8.517393171418904,
                4: -0.46985367797961614,
                5: 8.517393171418904,
                6: -0.8752354701118938,
                7: -0.46985367797961614,
                8: -0.8752354701118938,
            },
        }
    )

    pd.testing.assert_frame_equal(new_X, expected, check_less_precise=3)
