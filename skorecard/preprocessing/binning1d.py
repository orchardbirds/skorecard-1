import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from ..utils.exceptions import DimensionalityError
from probatus.binning import SimpleBucketer, AgglomerativeBucketer, QuantileBucketer, TreeBucketer


class BucketTransformer(BaseEstimator, TransformerMixin):
    """Base class for the below Bucket transformer methodologies using the Bucketers in the Probatus package."""

    def __init__(self, **kwargs):
        """Initialise with empty bucket dictionary to which we add our Bucketer(s) in self.fit()."""
        self.fitted = False
        self.BucketDict = {}
        self.kwargs = kwargs

    def _assert_bin_count_int(self):
        """If the bin_count is given as an int, then we turn it into a list."""
        if type(self.bin_count) != int:
            raise AttributeError("bin_count must be int")

    def _assert_1d_array(self, X, y):

        correct_X_shape = (X.ndim == 2 and X.shape[1] == 1) or X.ndim == 1

        if not correct_X_shape:
            raise DimensionalityError(f"The feature must be one-dimensional: X shape is {X.shape}")

        if y is not None:
            correct_y_shape = (y.ndim == 2 and y.shape[1] == 1) or y.ndim == 1

            if not correct_y_shape:
                raise DimensionalityError(f"The target must be one-dimensional: y shape is {y.shape}")
            y = y.copy()
            y = y.reshape(-1,)

        X = X.copy()
        X = X.reshape(-1,)

        return X, y

    def fit(self, X, y=None):
        """Fits the relevant Probatus bucket onto the numerical array.

        Args:
            X (np.array): The numerical data on which we wish to fit our BucketTransformer

        Returns:
            self (object): Fitted transformer
        """
        X = X.copy()
        X, y = self._assert_1d_array(X, y)

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

        X = np.digitize(X, self.boundaries[1:], right=True)

        return X.astype(int)

    def predict(self, X, y=None):
        """Applies the transform method. To be used for the grid searches.

        Args:
            X (np.array): The numerical data which will be transformed into the corresponding buckets

        Returns:
            np.array of the transformed X, such that the values of X are replaced by the corresponding bucket numbers
        """
        return self.transform(X, y)


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
        self._assert_bin_count_int()
        self.method = "Simple"

    def _fit(self, X, y=None):
        """Fits the Simple Probatus bucket onto the numerical array.

        Args:
            X (np.array): The numerical data on which we wish to fit our SimpleBucketTransformer

        Returns:
            self (object): Fitted transformer
        """
        self.Bucketer = SimpleBucketer(bin_count=self.bin_count)
        self.BucketDict["SimpleBucketer"] = self.Bucketer.fit(X)
        self.boundaries = self.BucketDict["SimpleBucketer"].boundaries


class AgglomerativeBucketTransformer(BucketTransformer):
    """Bucket transformer implementing the Agglomerative Bucketer in the Probatus package."""

    def __init__(self, bin_count):
        """Initialise BucketTransformer using Agglomerative Probatus Bucketer.

        Args:
            bin_count (int/list): How many bins we wish to split our data into. Required for each Probatus Bucket method
        """
        super().__init__()
        self.bin_count = bin_count
        self._assert_bin_count_int()
        self.method = "Agglomerative"

    def _fit(self, X, y=None):
        """Fits the Agglomerative Probatus bucket onto the numerical array.

        Args:
            X (np.array): The numerical data on which we wish to fit our AgglomerativeBucketTransformer

        Returns:
            self (object): Fitted transformer
        """
        self.Bucketer = AgglomerativeBucketer(bin_count=self.bin_count)
        self.BucketDict["AgglomerativeBucketer"] = self.Bucketer.fit(X)
        self.boundaries = self.BucketDict["AgglomerativeBucketer"].boundaries


class QuantileBucketTransformer(BucketTransformer):
    """Bucket transformer implementing the Quantile Bucketer in the Probatus package."""

    def __init__(self, bin_count):
        """Initialise BucketTransformer using Quantile Probatus Bucketer.

        Args:
            bin_count (int/list): How many bins we wish to split our data into. Required for each Probatus Bucket method
        """
        super().__init__()
        self.bin_count = bin_count
        self._assert_bin_count_int()
        self.method = "Quantile"

    def _fit(self, X, y=None):
        """Fits the Quantile Probatus bucket onto the numerical array.

        Args:
            X (np.array): The numerical data on which we wish to fit our QuantileBucketTransformer

        Returns:
            self (object): Fitted transformer
        """
        self.Bucketer = QuantileBucketer(bin_count=self.bin_count)
        self.BucketDict["QuantileBucketer"] = self.Bucketer.fit(X)
        self.boundaries = self.BucketDict["QuantileBucketer"].boundaries


class TreeBucketTransformer(BucketTransformer):
    """Bucket transformer implementing the Tree Bucketer in the Probatus package."""

    def __init__(self, **kwargs):
        """Initialise BucketTransformer using Tree Probatus Bucketer.

        Args:
            **kwargs: the keyword arguments passed to the Tree Probatus Bucketer
        """
        super().__init__(**kwargs)
        self.method = "Tree"
        self.Bucketer = TreeBucketer(**self.kwargs)

    def _fit(self, X, y=None):
        """Fits the Tree Probatus bucket onto the numerical array.

        Args:
            X (np.array): The numerical data on which we wish to fit our TreeBucketTransformer
            y (np.array): The binary target

        Returns:
            self (object): Fitted transformer
        """
        self.BucketDict["TreeBucketer"] = self.Bucketer.fit(X, y)
        self.boundaries = self.BucketDict["TreeBucketer"].boundaries

        return self

    def get_params(self, deep=True):
        """Return the parameters of the decision tree used in the Transfromer.

        Args:
            deep (boolean), required by the API.

        Returns:
            Decision Tree Parameteres (dict)

        """
        return self.Bucketer.tree.get_params(deep=deep)

    def set_params(self, **params):
        """Set the parameteres for the decision tree.

        Args:
            **params: (dict) parameteres for the decision tree

        """
        self.Bucketer.tree.set_params(**params)
        return self
