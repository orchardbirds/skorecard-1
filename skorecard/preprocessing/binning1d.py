"""
`skorecard` provides a collection of different Bucketer classes.

You can use these classes to find a binning on any `np.ndarray` or `pd.Series` object.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from probatus.binning import SimpleBucketer, AgglomerativeBucketer, QuantileBucketer, TreeBucketer

from skorecard.utils import DimensionalityError, assure_numpy_array
from skorecard.bucket_mapping import BucketMapping


class BucketTransformer(BaseEstimator, TransformerMixin):
    """Base class for the below Bucket transformer methodologies using the Bucketers in the Probatus package."""

    def __init__(self, **kwargs):
        """Initialise with empty bucket dictionary to which we add our Bucketer(s) in self.fit()."""
        self.fitted = False
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

        try:
            self.feature_name = X.name
        except AttributeError:
            self.feature_name = "unknown"

        X = assure_numpy_array(X)
        y = assure_numpy_array(y)
        X, y = self._assert_1d_array(X, y)

        self._fit(X, y)
        self.fitted = True

        return self

    def transform(self, X, y=None):
        """Transforms an array into the corresponding buckets fitted by the Transformer.

        Args:
            X (np.array): The numerical data which will be transformed into the corresponding buckets

        Returns:
            np.array of the transformed X, such that the values of X are replaced by the corresponding bucket numbers
        """
        return self.bucket_mapping.transform(X)

    def predict(self, X, y=None):
        """Applies the transform method. To be used for the grid searches.

        Args:
            X (np.array): The numerical data which will be transformed into the corresponding buckets

        Returns:
            np.array of the transformed X, such that the values of X are replaced by the corresponding bucket numbers
        """
        return self.transform(X, y)


class SimpleBucketTransformer(BucketTransformer):
    """Bucket transformer implementing the Simple Bucketer in the Probatus package.

    ```python
    from skorecard import datasets
    from skorecard.preprocessing import SimpleBucketTransformer

    X, y = datasets.load_uci_credit_card(return_X_y=True)
    bucketer = SimpleBucketTransformer(bin_count = 10)
    bucketer.fit_transform(X['LIMIT_BAL'])
    ```
    """

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
        self.Bucketer.fit(X)
        self.bucket_mapping = BucketMapping(
            feature_name=self.feature_name, type="numerical", map=self.Bucketer.boundaries, missing_bucket=None
        )


class AgglomerativeBucketTransformer(BucketTransformer):
    """Bucket transformer implementing the Agglomerative Bucketer in the Probatus package.

    ```python
    from skorecard import datasets
    from skorecard.preprocessing import AgglomerativeBucketTransformer

    X, y = datasets.load_uci_credit_card(return_X_y=True)
    bucketer = AgglomerativeBucketTransformer(bin_count = 10)
    bucketer.fit_transform(X['LIMIT_BAL'])
    ```
    """

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
        self.Bucketer.fit(X)
        self.bucket_mapping = BucketMapping(
            feature_name=self.feature_name, type="numerical", map=self.Bucketer.boundaries, missing_bucket=None
        )


class QuantileBucketTransformer(BucketTransformer):
    """Bucket transformer implementing the Quantile Bucketer in the Probatus package.

    ```python
    from skorecard import datasets
    from skorecard.preprocessing import QuantileBucketTransformer

    X, y = datasets.load_uci_credit_card(return_X_y=True)
    bucketer = QuantileBucketTransformer(bin_count = 10)
    bucketer.fit_transform(X['LIMIT_BAL'])
    ```
    """

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
        self.Bucketer.fit(X)
        self.bucket_mapping = BucketMapping(
            feature_name=self.feature_name, type="numerical", map=self.Bucketer.boundaries, missing_bucket=None
        )


class TreeBucketTransformer(BucketTransformer):
    """Bucket transformer implementing the Tree Bucketer in the Probatus package.

    ```python
    from skorecard import datasets
    from skorecard.preprocessing import TreeBucketTransformer

    X, y = datasets.load_uci_credit_card(return_X_y=True)
    bucketer = TreeBucketTransformer()
    bucketer.fit(X['LIMIT_BAL'], y)
    ```
    """

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
        self.Bucketer.fit(X, y)
        self.bucket_mapping = BucketMapping(
            feature_name=self.feature_name, type="numerical", map=self.Bucketer.boundaries, missing_bucket=None
        )

        return self

    def get_params(self, deep=True):
        """Return the parameters of the decision tree used in the Transfromer.

        Args:
            deep (bool): Make a deep copy or not, required by the API.

        Returns:
            (dict): Decision Tree Parameteres
        """
        return self.Bucketer.tree.get_params(deep=deep)

    def set_params(self, **params):
        """Set the parameteres for the decision tree.

        Args:
            **params: (dict) parameteres for the decision tree

        """
        self.Bucketer.tree.set_params(**params)
        return self


class CatBucketTransformer(BucketTransformer):
    """Bucket transformer for categorical features.

    ```python
    from skorecard import datasets
    from skorecard.preprocessing import CatBucketTransformer

    X, y = datasets.load_uci_credit_card(return_X_y=True)
    bucketer = CatBucketTransformer()
    bucketer.fit_transform(X['EDUCATION'])
    ```
    """

    def __init__(self, threshold_min=0.01):
        """Initialise Categorical Bucketer.

        Args:
            threshold_min (float): percentage. If the normalized value count is smaller than the threshold, the
                category is put into its own bin.

        """
        if not (0 <= threshold_min <= 1):
            raise ValueError("threshold_min must be between 0 and 1")

        super().__init__()
        self.method = "Categorical"
        self.threshold_min = threshold_min

    def _fit(self, X, y=None):
        """Fit the bucket transformer.

        Args:
            X (np.array): The numerical data on which we wish to fit our TreeBucketTransformer
            y (np.array): The binary target

        Returns:
            self (object): Fitted transformer
        """
        unique_categories, counts = np.unique(X, return_counts=True)
        counts = counts / X.shape[0]

        ix = np.argsort(counts)[::-1]
        sorted_cats = unique_categories[ix]
        sorted_counts = counts[ix]

        # find the unique values, sorted by counts
        if len(sorted_cats[sorted_counts > self.threshold_min]) > 0:
            map_dict = {val: index for index, val in enumerate(sorted_cats[sorted_counts > self.threshold_min])}
            # find all the counts below the threshold.
            # add them all in the latest category
            max_val = max(map_dict.values())
        else:
            map_dict = {}
            max_val = -1

        merged_bin = {i: max_val + 1 for i in sorted_cats[sorted_counts <= self.threshold_min]}
        map_dict.update(merged_bin)

        self.bucket_mapping = BucketMapping(
            feature_name=self.feature_name, type="categorical", map=map_dict, missing_bucket=None
        )

        return self

    def _transform(self, X, y=None):
        """Transform the input array based on the fitted categorical bucketer.

        Args:
            X (np.array): The categorical data which will be transformed into the corresponding buckets

        Returns:
            np.array of the transformed X, such that the values of X are replaced by the corresponding bucket numbers
        """
        X = X.copy()
        X = assure_numpy_array(X)
        X, y = self._assert_1d_array(X, y)

        return self.bucket_mapping.transform(X)
