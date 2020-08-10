import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from probatus.binning import SimpleBucketer, AgglomerativeBucketer, QuantileBucketer, TreeBucketer


class BucketTransformer(BaseEstimator, TransformerMixin):
    """Base class for the below Bucket transformer methodologies using the Bucketers in the Probatus package."""

    def __init__(self, **kwargs):
        """Initialise with empty bucket dictionary to which we add our Bucketer(s) in self.fit()."""
        self.fitted = False
        self.BucketDict = {}
        self.kwargs = kwargs

    def _check_list_size(self, X):
        """Checks that X and the list have the same number of features. We do not use this for the Manual Transformer.

        Args:
            X (np.array): the data on which we are fitting
        """
        if (self.method not in ["Manual", "Tree"]) and X.shape[1] != len(self.bin_count):
            raise ValueError(f"Length of bin_count ({len(self.bin_count)}) and shape of X ({X.shape[1]}) do not match!")

    def _enforce_bin_count_as_list(self):
        """If the bin_count is given as an int, then we turn it into a list."""
        if type(self.bin_count) == int:
            self.bin_count = [self.bin_count]
        elif type(self.bin_count) == float:
            raise AttributeError("bin_count must be int or list!")

    def _expand_single_entity_list(self, X):
        """If an int is passed for bin_count, then we want to ensure this is used for every feature."""
        if (self.method not in ["Manual", "Tree"]) and (len(self.bin_count) == 1):
            self.bin_count = np.repeat(self.bin_count, X.shape[1])

    def fit(self, X, y=None):
        """Fits the relevant Probatus bucket onto the numerical array.

        Args:
            X (np.array): The numerical data on which we wish to fit our BucketTransformer

        Returns:
            self (object): Fitted transformer
        """
        X = X.copy()
        if X.ndim == 1:
            X = np.expand_dims(X, 1)

        y = y.copy()
        if y.ndim == 1:
            y = np.expand_dims(y, 1)

        self._expand_single_entity_list(X)
        self._check_list_size(X)

        self._fit(X, y)
        self.fitted = True

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

        self._expand_single_entity_list(X)
        self._check_list_size(X)

        return self._transform(X, y)


class SimpleBucketTransformer(BucketTransformer):
    """Bucket transformer implementing the Simple Bucketer in the Probatus package."""

    def __init__(self, bin_count):
        """Initialise BucketTransformer using Simple Probatus Bucketer.

        Args:
            bin_count (int/list): How many bins we wish to split our data into for each feature.
                              Required for each Probatus Bucket method
        """
        super().__init__()
        self.bin_count = bin_count
        self._enforce_bin_count_as_list()
        self.method = "Simple"

    def _fit(self, X, y=None):
        """Fits the Simple Probatus bucket onto the numerical array.

        Args:
            X (np.array): The numerical data on which we wish to fit our SimpleBucketTransformer

        Returns:
            self (object): Fitted transformer
        """
        for i in range(X.shape[1]):
            self.Bucketer = SimpleBucketer(bin_count=self.bin_count[i])
            self.BucketDict[f"Feature_{i}"] = self.Bucketer.fit(X[:, i])

    def _transform(self, X, y=None):
        """Transforms a numerical array into its corresponding buckets using the fitted Simple Probatus Bucketer.

        Args:
            X (np.array): The numerical data which will be transformed into the corresponding buckets

        Returns:
            np.array of the transformed X, such that the values of X are replaced by the corresponding bucket numbers
        """
        for i in range(X.shape[1]):
            X[:, i] = np.digitize(X[:, i], self.BucketDict[f"Feature_{i}"].boundaries[1:], right=True,)

        return X


class AgglomerativeBucketTransformer(BucketTransformer):
    """Bucket transformer implementing the Agglomerative Bucketer in the Probatus package."""

    def __init__(self, bin_count):
        """Initialise BucketTransformer using Agglomerative Probatus Bucketer.

        Args:
            bin_count (int/list): How many bins we wish to split our data into. Required for each Probatus Bucket method
        """
        super().__init__()
        self.bin_count = bin_count
        self._enforce_bin_count_as_list()
        self.method = "Agglomerative"

    def _fit(self, X, y=None):
        """Fits the Agglomerative Probatus bucket onto the numerical array.

        Args:
            X (np.array): The numerical data on which we wish to fit our AgglomerativeBucketTransformer

        Returns:
            self (object): Fitted transformer
        """
        for i in range(X.shape[1]):
            self.Bucketer = AgglomerativeBucketer(bin_count=self.bin_count[i])
            self.BucketDict[f"Feature_{i}"] = self.Bucketer.fit(X[:, i])

    def _transform(self, X, y=None):
        """Transforms a numerical array into its corresponding buckets using the fitted Agglomerative Probatus Bucketer.

        Args:
            X (np.array): The numerical data which will be transformed into the corresponding buckets

        Returns:
            np.array of the transformed X, such that the values of X are replaced by the corresponding bucket numbers
        """
        for i in range(X.shape[1]):
            X[:, i] = np.digitize(X[:, i], self.BucketDict[f"Feature_{i}"].boundaries[1:], right=True,)

        return X


class QuantileBucketTransformer(BucketTransformer):
    """Bucket transformer implementing the Quantile Bucketer in the Probatus package."""

    def __init__(self, bin_count):
        """Initialise BucketTransformer using Quantile Probatus Bucketer.

        Args:
            bin_count (int/list): How many bins we wish to split our data into. Required for each Probatus Bucket method
        """
        super().__init__()
        self.bin_count = bin_count
        self._enforce_bin_count_as_list()
        self.method = "Quantile"

    def _fit(self, X, y=None):
        """Fits the Quantile Probatus bucket onto the numerical array.

        Args:
            X (np.array): The numerical data on which we wish to fit our QuantileBucketTransformer

        Returns:
            self (object): Fitted transformer
        """
        for i in range(X.shape[1]):
            self.Bucketer = QuantileBucketer(bin_count=self.bin_count[i])
            self.BucketDict[f"Feature_{i}"] = self.Bucketer.fit(X[:, i])

    def _transform(self, X, y=None):
        """Transforms a numerical array into its corresponding buckets using the fitted Quantile Probatus Bucketer.

        Args:
            X (np.array): The numerical data which will be transformed into the corresponding buckets

        Returns:
            np.array of the transformed X, such that the values of X are replaced by the corresponding bucket numbers
        """
        for i in range(X.shape[1]):
            X[:, i] = np.digitize(X[:, i], self.BucketDict[f"Feature_{i}"].boundaries[1:], right=True,)

        return X


class ManualBucketTransformer(BucketTransformer):
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

    def _fit(self, X, y=None):
        """As the boundaries are already defined here, we do not need a fit function, and hence we leave this empty."""
        return self

    def _transform(self, X, y=None):
        """Transforms a numerical array into its corresponding buckets using the user-defined boundaries.

        Args:
            X (np.array): The numerical data which will be transformed into the corresponding buckets

        Returns:
            np.array of the transformed X, such that the values of X are replaced by the corresponding bucket numbers
        """
        for i, v in enumerate(self.boundary_dict):
            X[:, i] = np.digitize(X[:, i], self.boundary_dict[v][1:], right=True,)
        return X


class TreeBucketTransformer(BucketTransformer):
    """Bucket transformer implementing the Tree Bucketer in the Probatus package."""

    def __init__(self, **kwargs):
        """Initialise BucketTransformer using Tree Probatus Bucketer.

        Args:
            bin_count (int/list): How many bins we wish to split our data into. Required for each Probatus Bucket method
        """
        super().__init__(**kwargs)
        self.method = "Tree"

    def _fit(self, X, y=None):
        """Fits the Tree Probatus bucket onto the numerical array.

        Args:
            X (np.array): The numerical data on which we wish to fit our TreeBucketTransformer
            y (np.array): The binary target

        Returns:
            self (object): Fitted transformer
        """
        for i in range(X.shape[1]):
            print(X.shape)
            self.Bucketer = TreeBucketer(**self.kwargs)
            self.BucketDict[f"Feature_{i}"] = self.Bucketer.fit(X[:, i], y)

        return self

    def _transform(self, X, y=None):
        """Transforms a numerical array into its corresponding buckets using the fitted Tree Probatus Bucketer.

        Args:
            X (np.array): The numerical data which will be transformed into the corresponding buckets

        Returns:
            np.array of the transformed X, such that the values of X are replaced by the corresponding bucket numbers
        """
        for i in range(X.shape[1]):
            X[:, i] = np.digitize(X[:, i], self.BucketDict[f"Feature_{i}"].boundaries[1:], right=True,)

        return X
