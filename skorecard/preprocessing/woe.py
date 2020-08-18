import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class WOETransformer(BaseEstimator, TransformerMixin):
    """Transformer to map the bucketed features to their Weight of Evidence estimation."""

    def __init__(self, epsilon=0.0001):
        """Constructor for WOETransformer.

        Args:
            epsilon (float): Amount to be added to relative counts in order to avoid division by zero in the WOE
            calculation.
        """
        self.epsilon = epsilon
        self.isFitted = False

    def fit(self, X, y):
        """Calculate the WOE for every column.

        Args:
            X (np.array): (binned) features
            y (np.array): target
        """
        self.woe_dicts = list()

        for i in range(X.shape[1]):
            bins, woe_values, counts_0, counts_1 = woe_1d(X[:, i], y, self.epsilon)
            self.woe_dicts.append(make_dict(bins, woe_values))

        self.isFitted = True
        return self

    def transform(self, X, y=None):
        """Apply the WOE to the feature set X.

        Args:
            X (np.array): (binned) features
            y (np.array): target, default is None. Not used, only here for API consistency

        Returns: (np.array) transformed features, mapped from bin indices to Weight of Evidence

        """
        X_woe = -9999.0 * np.ones(shape=X.shape)
        for i in range(X.shape[1]):
            X_woe[:, i] = _map_bins_to_woe(X[:, i], self.woe_dicts[i])

        return X_woe


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


def make_dict(bins, woe_values):
    """Convert the bins and woe_values array to a mapping dictionary.

    Args:
        bins (np.array): indices of unique values of X
        woe_values (np.array):  calculated weight of evidence for every unique bin

    Returns: (dict) dictionary with keys being the bins, and the woe_values the weight-of-evidences per bins
    """
    return {bin_: woe for bin_, woe in zip(bins, woe_values)}


def _map_bins_to_woe(X_bins, map_dict):
    """Map the bins to the equivalent weight of evidence, as per the map_dict.

    Args:
        X_bins (np.array): 1d array, (binned) feature
        map_dict (dict): dictionary with maps to be applied to X_bins as a map

    Returns: (np.array), with same shape as X_bins, where every unique value is converted to the WOE
    """
    return np.vectorize(map_dict.get)(X_bins)
