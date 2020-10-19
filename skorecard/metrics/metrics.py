import numpy as np

from sklearn.metrics import make_scorer


def woe_1d(x, y, epsilon=0.0001):
    """Compute the weight of evidence on a 1-dimensional array.

    Args:
        x (np.array): 1d array, (binned) feature
        y (np.array): target
        epsilon (float): Amount to be added to relative counts in order to avoid division by zero in the WOE
            calculation.

    Returns: (tuple of numpy.arrays)
        - bins: indices of unique values of X
        - woe_values: calculated weight of evidence for every unique bin
        - counts_0: count of entries per bin where y==0
        - counts_1: count of entries per bin where y==1
    """
    if 0 not in x:
        raise ValueError("Array x must contain an index 0")
    if len(np.unique(y)) != 2:
        raise ValueError("y must contain a binary target")

    x_0 = x[y == 0]
    x_1 = x[y == 1]

    total_0 = x_0.shape[0]
    total_1 = x_1.shape[0]

    bins = np.unique(x)
    counts_0 = np.bincount(x_0, minlength=len(bins))
    counts_1 = np.bincount(x_1, minlength=len(bins))

    woe_num = (counts_0 / total_0) + epsilon
    woe_denom = (counts_1 / total_1) + epsilon

    # Make sure to give informative error when dividing by zero error occurs
    msg = """
    One of the unique values in X has no occurances of the %s class.
    Set epsilon to a very small value, or use a more coarse binning.
    """
    if any(woe_num == 0):
        raise ZeroDivisionError(msg % "negative")
    if any(woe_denom == 0):
        raise ZeroDivisionError(msg % "positive")

    woe_values = np.log(woe_num / woe_denom)

    return bins, woe_values, counts_0, counts_1


def _IV_score(y_test, y_pred):
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


@make_scorer
def IV_scorer(y_test, y_pred):
    """Decorated version. Makse the  IV score usable for sklearn grid search pipelines.

    Using the unique values in y_pred, calculates the information value for the specific np.array.

    Args:
        y_test: (np.array), binary features, target
        y_pred: (np.array), predictions, indices of the buckets where the IV should be computed

    Returns:
        iv (float): information value

    """
    return _IV_score(y_test, y_pred)
