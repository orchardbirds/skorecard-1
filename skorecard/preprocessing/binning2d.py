import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class ManualBucketTransformer(BaseEstimator, TransformerMixin):
    """Bucket transformer implementing user-defined boundaries."""

    def __init__(self, boundary_dict):
        """Initialise the user-defined boundaries with a dictionary.

        Args:
            boundary_dict (dict): Contains the feature column number and boundaries defined for this feature.
                                  For example, the boundary_dict {1: [0, 5, 10],
                                                                  3: [6, 8]}
                                  means we apply boundaries [0, 5, 10] to column 1 and boundaries [6, 8] to column 3
        """
        super().__init__()
        self.boundary_dict = boundary_dict
        self.method = "Manual"

    def fit(self, X):
        """Dummy fit function, needed to comply to the sklearn API. Does nothing.

        Args:
            X: (np.array): The numerical data which will be transformed into the corresponding buckets
        """
        return self

    def transform(self, X, y=None):
        """Transforms a numerical array into its corresponding buckets using the fitted Probatus bucket methodology.

        Args:
            X (np.array): The numerical data which will be transformed into the corresponding buckets

        Returns:
            np.array of the transformed X, such that the values of X are replaced by the corresponding bucket numbers
        """
        X = X.copy()
        if X.ndim == 1:
            X = np.expand_dims(X, 1)

        for ix in self.boundary_dict.keys():
            X[:, ix] = np.digitize(X[:, ix], self.boundary_dict[ix], right=True)

        return X
