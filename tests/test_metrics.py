import numpy as np
import pytest
from skorecard.metrics import metrics


@pytest.fixture()
def X_y():
    """Set of X,y for testing the transformers."""
    X = np.array([[0, 1], [1, 0], [0, 0], [3, 2], [0, 1], [1, 2], [2, 0], [2, 1], [0, 0]], np.int32,)
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1])

    return X, y


def test_iv_on_array(X_y):
    """Test the IV calculation for two arrays."""
    X, y = X_y

    np.testing.assert_almost_equal(metrics._IV_score(y, X[:, 0]), 5.307, decimal=2)

    np.testing.assert_almost_equal(metrics._IV_score(y, X[:, 1]), 4.635, decimal=2)
