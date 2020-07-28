import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from probatus.binning import SimpleBucketer, AgglomerativeBucketer, QuantileBucketer


class BucketTransformer(BaseEstimator, TransformerMixin):
    """Base class for the below Bucket transformer methodologies using the Bucketers in the Probatus package."""

    def __init__(self):
        """Initialise."""
        self.fitted = False

    def fit(self, X, y=None):
        """Fits the relevant Probatus bucket onto the numerical array.

        Args:
            X (np.array): The numerical data on which we wish to fit our BucketTransformer

        Returns:
            self (object): Fitted transformer
        """
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
        return self._transform(X, y)


class SimpleBucketTransformer(BucketTransformer):
    """Bucket transformer implementing the Simple Bucketer in the Probatus package."""

    def __init__(self, bin_count):
        """Initialise BucketTransformer using Simple Probatus Bucketer.

        Args:
            bin_count (int): How many bins we wish to split our data into. Required for each Probatus Bucket method
        """
        super().__init__()
        self.bin_count = bin_count

    def _fit(self, X, y=None):
        """Fits the Simple Probatus bucket onto the numerical array.

        Args:
            X (np.array): The numerical data on which we wish to fit our SimpleBucketTransformer

        Returns:
            self (object): Fitted transformer
        """
        self.Bucketer = SimpleBucketer(bin_count=self.bin_count)
        self.Bucketer.fit(X)

    def _transform(self, X, y=None):
        """Transforms a numerical array into its corresponding buckets using the fitted Simple Probatus Bucketer.

        Args:
            X (np.array): The numerical data which will be transformed into the corresponding buckets

        Returns:
            np.array of the transformed X, such that the values of X are replaced by the corresponding bucket numbers
        """
        return np.digitize(X, self.Bucketer.boundaries[1:], right=True)


class AgglomerativeBucketTransformer(BucketTransformer):
    """Bucket transformer implementing the Agglomerative Bucketer in the Probatus package."""

    def __init__(self, bin_count):
        """Initialise BucketTransformer using Agglomerative Probatus Bucketer.

        Args:
            bin_count (int): How many bins we wish to split our data into. Required for each Probatus Bucket method
        """
        super().__init__()
        self.bin_count = bin_count

    def _fit(self, X, y=None):
        """Fits the Agglomerative Probatus bucket onto the numerical array.

        Args:
            X (np.array): The numerical data on which we wish to fit our AgglomerativeBucketTransformer

        Returns:
            self (object): Fitted transformer
        """
        self.Bucketer = AgglomerativeBucketer(bin_count=self.bin_count)
        self.Bucketer.fit(X)

    def _transform(self, X, y=None):
        """Transforms a numerical array into its corresponding buckets using the fitted Agglomerative Probatus Bucketer.

        Args:
            X (np.array): The numerical data which will be transformed into the corresponding buckets

        Returns:
            np.array of the transformed X, such that the values of X are replaced by the corresponding bucket numbers
        """
        return np.digitize(X, self.Bucketer.boundaries[1:], right=True)


class QuantileBucketTransformer(BucketTransformer):
    """Bucket transformer implementing the Quantile Bucketer in the Probatus package."""

    def __init__(self, bin_count):
        """Initialise BucketTransformer using Quantile Probatus Bucketer.

        Args:
            bin_count (int): How many bins we wish to split our data into. Required for each Probatus Bucket method
        """
        super().__init__()
        self.bin_count = bin_count

    def _fit(self, X, y=None):
        """Fits the Quantile Probatus bucket onto the numerical array.

        Args:
            X (np.array): The numerical data on which we wish to fit our QuantileBucketTransformer

        Returns:
            self (object): Fitted transformer
        """
        self.Bucketer = QuantileBucketer(bin_count=self.bin_count)
        self.Bucketer.fit(X)

    def _transform(self, X, y=None):
        """Transforms a numerical array into its corresponding buckets using the fitted Quantile Probatus Bucketer.

        Args:
            X (np.array): The numerical data which will be transformed into the corresponding buckets

        Returns:
            np.array of the transformed X, such that the values of X are replaced by the corresponding bucket numbers
        """
        return np.digitize(X, self.Bucketer.boundaries[1:], right=True)
