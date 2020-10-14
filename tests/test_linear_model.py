import numpy as np

from skorecard import datasets
from skorecard.linear_model import LogisticRegression

import pytest


@pytest.fixture()
def X_y():
    """Generate dataframe."""
    X, y = datasets.load_uci_credit_card(return_X_y=True)
    return X, y


def test_output_dimensions():
    """Test the dimensions of the new attributes."""
    shape_features = (10, 3)

    X = np.random.uniform(size=shape_features)
    y = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1])

    lr = LogisticRegression(fit_intercept=True).fit(X, y)

    assert lr.p_val_coef_.shape[1] == shape_features[1]
    assert lr.z_coef_.shape[1] == shape_features[1]
    assert lr.std_err_coef_.shape[1] == shape_features[1]
    assert not np.isnan(lr.p_val_intercept_)
    assert not np.isnan(lr.z_intercept_)
    assert not np.isnan(lr.std_err_intercept_)

    lr = LogisticRegression(fit_intercept=False).fit(X, y)

    assert lr.p_val_coef_.shape[1] == shape_features[1]
    assert lr.z_coef_.shape[1] == shape_features[1]
    assert lr.std_err_coef_.shape[1] == shape_features[1]
    assert np.isnan(lr.p_val_intercept_)
    assert np.isnan(lr.z_intercept_)
    assert np.isnan(lr.std_err_intercept_)


def test_results(X_y):
    """Test the actual p-values."""
    expected_approx_p_val_coef_ = np.array([[1.0, 1.0, 0.0, 0.8425]])

    lr = LogisticRegression(fit_intercept=True, penalty="none").fit(*X_y)

    np.testing.assert_array_almost_equal(lr.p_val_coef_, expected_approx_p_val_coef_, decimal=3)