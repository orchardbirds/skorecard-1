import numpy as np
from skorecard.preprocessing import WOETransformer
import pytest


@pytest.fixture()
def X_y():
    """Set of X,y for testing the transformers."""
    X = np.array([[0, 1], [1, 0], [0, 0], [3, 2], [0, 1], [1, 2], [2, 0], [2, 1], [0, 0]], np.int32)
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1])

    return X, y


@pytest.fixture()
def X_y_2():
    """Set of X,y for testing the transformers.

    In the first column, bucket 3 is not present in class 1.
    """
    X = np.array([[0, 1], [1, 0], [0, 0], [3, 2], [0, 1], [1, 2], [2, 0], [2, 1], [0, 0]], np.int32)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])

    return X, y


def test_woe_transformed_dimensions(X_y):
    """Tests that the total number of unique WOEs matches the unique number of bins in X."""
    X, y = X_y

    X_WOE = WOETransformer().fit_transform(X, y)

    assert X_WOE.shape == X.shape
    assert len(np.unique(X[:, 0])) == len(np.unique(X_WOE[:, 0]))
    assert len(np.unique(X[:, 1])) == len(np.unique(X_WOE[:, 1]))


def test_missing_bucket(X_y_2):
    """Tests that the total number of unique WOEs matches the unique number of bins in X."""
    X, y = X_y_2

    X_WOE = WOETransformer().fit_transform(X, y)

    assert X_WOE.shape == X.shape
    assert len(np.unique(X[:, 0])) == len(np.unique(X_WOE[:, 0]))
    # because class 1 will have zero counts, the WOE transformer will divide by the value of epsilon, avoinding infinite
    # numbers
    assert np.abs(X_WOE[3, 0]) != np.inf
    assert all(X_WOE[[6, 7], 0] != np.array([-np.inf, -np.inf]))

    # If epsilon is set to zero, expect a Division By Zero exception
    X_WOE = WOETransformer(epsilon=0).fit_transform(X, y)

    # The third element of the array is 3, which is not present in class 1, hence expecting a positive infinite
    assert X_WOE[3, 0] == np.inf

    # The sizt and sevet element of the array is 2, which is not present in class 0, hence expecting a positive infinite
    assert all(X_WOE[[6, 7], 0] == np.array([-np.inf, -np.inf]))
