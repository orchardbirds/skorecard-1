import numpy as np

from sklearn.metrics import make_scorer


def woe_1d(X, y, epsilon=0.0001):
    """Compute the weight of evidence on a 1-dimensional array.

    Args:
        X (np.array): 1d array, (binned) feature
        y (np.array): target
        epsilon (float): Amount to be added to relative counts in order to avoid division by zero in the WOE
            calculation.

    Returns: (tuple of numpy.arrays)
        - bins: indices of unique values of X
        - woe_values: calculated weight of evidence for every unique bin
        - counts_0: count of entries per bin where y==0
        - counts_1: count of entries per bin where y==1
    """
    if 0 not in X:
        raise ValueError("Array must contain an index 0")
    X_0 = X[y == 0]
    X_1 = X[y == 1]

    total_0 = X_0.shape[0]
    total_1 = X_1.shape[0]

    bins = np.unique(X)
    counts_0 = np.bincount(X_0, minlength=len(bins))
    counts_1 = np.bincount(X_1, minlength=len(bins))

    woe_num = (counts_0 / total_0) + epsilon
    woe_denom = (counts_1 / total_1) + epsilon

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
