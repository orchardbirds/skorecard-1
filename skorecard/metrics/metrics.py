from skorecard.preprocessing.woe import woe_1d
from sklearn.metrics import make_scorer


@make_scorer
def IV_score(y_test, y_pred):
    """Using the unique values in y_pred, calculates the information value for the specific np.array.

    Args:
        y_test: (np.array), binary features, target
        y_pred: (np.array), predictions, indices of the buckets where the IV should be computed

    Returns:
        iv (float): information value

    """
    dummy, woes, c_0, c_1 = woe_1d(y_pred, y_test)

    dist_0 = c_0 / c_0.sum()
    dist_1 = c_1 / c_1.sum()

    iv = ((dist_0 - dist_1) * woes).sum()

    return iv
