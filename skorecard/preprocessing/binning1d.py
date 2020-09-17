import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from ..utils import DimensionalityError, assure_numpy_array
from probatus.binning import SimpleBucketer, AgglomerativeBucketer, QuantileBucketer, TreeBucketer


class BucketTransformer(BaseEstimator, TransformerMixin):
    """Base class for the below Bucket transformer methodologies using the Bucketers in the Probatus package."""

    def __init__(self, infinite_edges=False, **kwargs):
        """Initialise with empty bucket dictionary to which we add our Bucketer(s) in self.fit()."""
        self.fitted = False
        self.BucketDict = {}
        self.infinite_edges = infinite_edges
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

        X = assure_numpy_array(X)
        y = assure_numpy_array(y)

        X, y = self._assert_1d_array(X, y)

        self._fit(X, y)
        if self.infinite_edges:
            self.boundaries[0] = -np.inf
            self.boundaries[-1] = np.inf
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

        if self.method == "Categorical":
            return self._transform(X)
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

    def __init__(self, bin_count, infinite_edges=False):
        """Initialise BucketTransformer using Simple Probatus Bucketer.

        Args:
            bin_count (int/list): How many bins we wish to split our data into for each feature.
                              Required for each Probatus Bucket method
            infinite_edges (boolean): flag to set the edges of the bins to infinite values.
        """
        super().__init__(infinite_edges=infinite_edges)
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

    def __init__(self, bin_count, infinite_edges=False):
        """Initialise BucketTransformer using Agglomerative Probatus Bucketer.

        Args:
            bin_count (int/list): How many bins we wish to split our data into. Required for each Probatus Bucket method
            infinite_edges (boolean): flag to set the edges of the bins to infinite values.
        """
        super().__init__(infinite_edges=infinite_edges)
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

    def __init__(self, bin_count, infinite_edges=False):
        """Initialise BucketTransformer using Quantile Probatus Bucketer.

        Args:
            bin_count (int/list): How many bins we wish to split our data into. Required for each Probatus Bucket method
             infinite_edges (boolean): flag to set the edges of the bins to infinite values.
        """
        super().__init__(infinite_edges=infinite_edges)
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

    def __init__(self, infinite_edges=False, **kwargs):
        """Initialise BucketTransformer using Tree Probatus Bucketer.

        Args:
            infinite_edges (boolean): flag to set the edges of the bins to infinite values.
            **kwargs: the keyword arguments passed to the Tree Probatus Bucketer
        """
        super().__init__(infinite_edges, **kwargs)
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


class CatBucketTransformer(BucketTransformer):
    """Bucket transformer for categorical features."""

    def __init__(self, threshold_min=0.01, threshold_max=0.2, epsilon=0.05):
        """Initialise Categorical Bucketer.

        Args:
            threshold_min (float): percentage. If the normalized value count is smaller than the threshold, the
            category is put into its own bin.
            threshold_max (float): percentage. If the normalized value count is larger than the threshold, the
            category is put into its own bin.
            All categories between threshold_min and threshold_max get lumped together into 1 bin.
            epsilon (float): Between 0 and 1. After the categories have been bucketed according to value counts, we
            calculate the default rate per bin. Any default rates within epsilon are bucketed together.
        """
        if (epsilon < 0) | (epsilon > 1):
            raise ValueError("epsilon must be between 0 and 1")
        if (threshold_min < 0) | (threshold_min > 1):
            raise ValueError("threshold_min must be between 0 and 1")
        if (threshold_max < 0) | (threshold_max > 1):
            raise ValueError("threshold_max must be between 0 and 1")
        if threshold_min > threshold_max:
            raise AttributeError("threshold_min must be less than threshold_max")
        super().__init__()
        self.method = "Categorical"
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.epsilon = epsilon

    def _bucket_on_value_counts(self, X):
        """Calculates the normalized value counts for each category in X and bins based on the percentages.

        Args:
            X: numpy array of the categorical column

        Returns:
            Bucketed X based on normalized value counts
        """
        X = X.copy()
        X = np.array(X, dtype="object")

        self._category_to_bucket_dict = {}
        bucket = 0
        unique_categories, counts = np.unique(X, return_counts=True)

        # Normalize counts
        counts = counts / X.shape[0]

        # Go through every unique category and bucket everything
        for i in range(len(unique_categories)):
            category = unique_categories[i]
            if counts[i] > self.threshold_max:
                # We have a bucket!
                X[X == category] = f"value_count_bucket_{bucket}"
                self._category_to_bucket_dict[f"original_category_{category}"] = f"value_count_bucket_{bucket}"

                bucket += 1
            elif counts[i] < self.threshold_min:
                # Not enough in this category
                X[X == category] = "between_thresholds"
                self._category_to_bucket_dict[f"original_category_{category}"] = "between_thresholds"
            else:
                # Todo: We might want to do something clever here, as we have threshold_min <= x <= threshold_max
                # Not enough in this category
                X[X == category] = "between_thresholds"
                self._category_to_bucket_dict[f"original_category_{category}"] = "between_thresholds"

        # Final bucket for the small values
        X[X == "between_thresholds"] = f"value_count_bucket_{bucket}"

        # Final bucket for the small values in dict
        for _, key in enumerate(self._category_to_bucket_dict):
            if self._category_to_bucket_dict[key] == "between_thresholds":
                self._category_to_bucket_dict[key] = f"value_count_bucket_{bucket}"

        return X

    def _calculate_default_rate(self, X, y):
        """Calculates the default rate for the already-bucketed categorical column.

        Args:
            X: numpy array of the bucketed column, i.e. X = self._bucket_on_value_counts()
            y: the target, used to calculate the default rate

        Returns:
            default_rates (dict): The key is the bucket, the value is the default rate.
        """
        default_rates = {}
        unique_categories, counts = np.unique(X, return_counts=True)
        ar = np.column_stack((X, y))

        # Go through bucketed categories, calculate the default rate per bucket
        for i in range(len(unique_categories)):
            category = unique_categories[i]
            bucket = ar[ar[:, 0] == category]
            default_rate = bucket[bucket[:, 1] == 1].shape[0] / bucket.shape[0]
            default_rates[category] = default_rate

        return default_rates

    def _group_default_rates(self):
        """Groups the default rates if they are within epsilon.

        The grouping is done downwards, such that if bucket 1 and bucket 2 are within episilon, bucket 2 will change
        its default rate.

        Returns:
            grouped_default_rates (dict): A dictionary with modified values if the default rates of the buckets are
            within epsilon.
        """
        default_rates = self._default_rates.copy()
        _grouped_default_rates = self._default_rates.copy()

        for i in range(len(default_rates)):
            for j in range(i, len(default_rates)):
                key_a = list(default_rates.keys())[i]
                key_b = list(_grouped_default_rates.keys())[j]
                if key_a == key_b:
                    continue
                if np.abs(default_rates[key_a] - _grouped_default_rates[key_b]) <= self.epsilon:
                    _grouped_default_rates[key_b] = _grouped_default_rates[key_a]

        return _grouped_default_rates

    def _create_mapping_dict(self):
        """Creates a mapping dictionary for the user.

        This enables them to see the final buckets that the original categories are mapped to.

        For example:
          {"original_category_0": 6}.

        This means that the original categorical column contained a value 0, which is now mapped to bucket 6 when
        bucketed by default rates.
        """
        self.mapping_dict = {}

        for key in self._category_to_bucket_dict.keys():
            self.mapping_dict[key] = self._default_rate_bins[
                self._grouped_default_rates[self._category_to_bucket_dict[key]]
            ]

    def _bucket_on_default_rates(self, X):
        """Buckets X based on the default rate dictionary.

        Currently we use a simple (read: dumb) way of bucketing the default rates:
        we lump everything between 0-0.05%, 0.05-0.1%, ..., 0.95-1.0% together.

        Args:
            X: Numpy array

        Returns:
            X bucketed on default rates.
        """
        # First we must bucket by value_counts
        X = self._bucket_on_value_counts(X)
        X_dr = X.copy()
        unique_categories, counts = np.unique(X, return_counts=True)

        # Group default rates
        self._grouped_default_rates = self._group_default_rates()

        # Convert to default rates
        for i in range(len(unique_categories)):
            category = unique_categories[i]
            X_dr[X_dr == category] = self._grouped_default_rates[category]

        # Sort values in order
        unique_default_bins = sorted(np.unique(X_dr))

        # convert to bins on grouped default rate
        self._default_rate_bins = {}
        for i in range(len(unique_default_bins)):
            unique_default_bin = unique_default_bins[i]
            X_dr[X_dr == unique_default_bin] = i
            self._default_rate_bins[unique_default_bin] = i

        self._create_mapping_dict()

        return X_dr

    def _fit(self, X, y=None):
        """Calculates the default rates for each bucket.

        Args:
            X: Numpy array of categorical column.
            y: Target, used for calculating default rates.
        """
        X = self._bucket_on_value_counts(X)
        self._default_rates = self._calculate_default_rate(X, y)

        return self

    def _transform(self, X, y=None):
        """Transforms the categorical column to a bucket using the default rates."""
        return self._bucket_on_default_rates(X)
