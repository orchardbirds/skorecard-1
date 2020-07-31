import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from probatus.binning import SimpleBucketer, AgglomerativeBucketer, QuantileBucketer
import yaml


class BucketTransformer(BaseEstimator, TransformerMixin):
    """Base class for the below Bucket transformer methodologies using the Bucketers in the Probatus package."""

    def __init__(self):
        """Initialise with empty bucket dictionary to which we add our Bucketer(s) in self.fit()."""
        self.fitted = False
        self.BucketDict = {}

    def _check_dict_size(self, X):
        """Checks that X and the dictionary have the same number of features.

        Args:
            X (np.array): the data on which we are fitting
        """
        if X.shape[1] != len(self.bin_count):
            raise ValueError(f"Length of bin_count ({len(self.bin_count)}) and shape of X ({X.shape[1]}) do not match!")

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

        if isinstance(self.bin_count, dict):
            self._check_dict_size(self, X)

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
        if isinstance(self.bin_count, dict):
            self._check_dict_size(self, X)

        X = X.copy()
        if X.ndim == 1:
            X = np.expand_dims(X, 1)

        return self._transform(X, y)

    def save(self, filename):
        """Save the self.BucketDict in a YAML file. A separate section is written per Bucket object in this dictionary.

        See docs/example.YAML for an example how this is saved

        Args:
            filename (str): Name of the YAML file that is saved
        """
        with open(f"{filename}.yaml", "w") as outfile:
            for i, v in enumerate(self.BucketDict):
                tmp_dict = {}
                tmp_dict["bin_count"] = self.BucketDict[v].bin_count
                tmp_dict["boundaries"] = self.BucketDict[v].boundaries.tolist()
                yaml.dump({v: tmp_dict}, outfile, default_flow_style=False)


class SimpleBucketTransformer(BucketTransformer):
    """Bucket transformer implementing the Simple Bucketer in the Probatus package."""

    def __init__(self, bin_count):
        """Initialise BucketTransformer using Simple Probatus Bucketer.

        Args:
            bin_count (int): How many bins we wish to split our data into. Required for each Probatus Bucket method
        """
        super().__init__()
        self.bin_count = bin_count
        self.method = "Simple"

    def _fit(self, X, y=None):
        """Fits the Simple Probatus bucket onto the numerical array.

        Args:
            X (np.array): The numerical data on which we wish to fit our SimpleBucketTransformer

        Returns:
            self (object): Fitted transformer
        """
        if isinstance(self.bin_count, dict):
            for i, v in enumerate(self.bin_count):
                self.Bucketer = SimpleBucketer(bin_count=self.bin_count[v])
                self.BucketDict[f"Feature_{v}"] = self.Bucketer.fit(X[:, i])

        else:
            for i in range(X.shape[1]):
                self.Bucketer = SimpleBucketer(bin_count=self.bin_count)
                self.BucketDict[f"Feature_{i}"] = self.Bucketer.fit(X[:, i])

    def _transform(self, X, y=None):
        """Transforms a numerical array into its corresponding buckets using the fitted Simple Probatus Bucketer.

        Args:
            X (np.array): The numerical data which will be transformed into the corresponding buckets

        Returns:
            np.array of the transformed X, such that the values of X are replaced by the corresponding bucket numbers
        """
        if isinstance(self.bin_count, dict):
            for i, v in enumerate(self.bin_count):
                X[:, i] = np.digitize(X[:, i], self.BucketDict[f"Feature_{v}"].boundaries[1:], right=True,)

        else:
            for i in range(X.shape[1]):
                X[:, i] = np.digitize(X[:, i], self.BucketDict[f"Feature_{i}"].boundaries[1:], right=True,)

        return X


class AgglomerativeBucketTransformer(BucketTransformer):
    """Bucket transformer implementing the Agglomerative Bucketer in the Probatus package."""

    def __init__(self, bin_count):
        """Initialise BucketTransformer using Agglomerative Probatus Bucketer.

        Args:
            bin_count (int): How many bins we wish to split our data into. Required for each Probatus Bucket method
        """
        super().__init__()
        self.bin_count = bin_count
        self.method = "Agglomerative"

    def _fit(self, X, y=None):
        """Fits the Agglomerative Probatus bucket onto the numerical array.

        Args:
            X (np.array): The numerical data on which we wish to fit our AgglomerativeBucketTransformer

        Returns:
            self (object): Fitted transformer
        """
        if isinstance(self.bin_count, dict):
            for i, v in enumerate(self.bin_count):
                self.Bucketer = AgglomerativeBucketer(bin_count=self.bin_count[v])
                self.BucketDict[f"Feature_{v}"] = self.Bucketer.fit(X[:, i])

        else:
            for i in range(X.shape[1]):
                self.Bucketer = AgglomerativeBucketer(bin_count=self.bin_count)
                self.BucketDict[f"Feature_{i}"] = self.Bucketer.fit(X[:, i])

    def _transform(self, X, y=None):
        """Transforms a numerical array into its corresponding buckets using the fitted Agglomerative Probatus Bucketer.

        Args:
            X (np.array): The numerical data which will be transformed into the corresponding buckets

        Returns:
            np.array of the transformed X, such that the values of X are replaced by the corresponding bucket numbers
        """
        if isinstance(self.bin_count, dict):
            for i, v in enumerate(self.bin_count):
                X[:, i] = np.digitize(X[:, i], self.BucketDict[f"Feature_{v}"].boundaries[1:], right=True,)

        else:
            for i in range(X.shape[1]):
                X[:, i] = np.digitize(X[:, i], self.BucketDict[f"Feature_{i}"].boundaries[1:], right=True,)

        return X


class QuantileBucketTransformer(BucketTransformer):
    """Bucket transformer implementing the Quantile Bucketer in the Probatus package."""

    def __init__(self, bin_count):
        """Initialise BucketTransformer using Quantile Probatus Bucketer.

        Args:
            bin_count (int): How many bins we wish to split our data into. Required for each Probatus Bucket method
        """
        super().__init__()
        self.bin_count = bin_count
        self.method = "Quantile"

    def _fit(self, X, y=None):
        """Fits the Quantile Probatus bucket onto the numerical array.

        Args:
            X (np.array): The numerical data on which we wish to fit our QuantileBucketTransformer

        Returns:
            self (object): Fitted transformer
        """
        if isinstance(self.bin_count, dict):
            for i, v in enumerate(self.bin_count):
                self.Bucketer = QuantileBucketer(bin_count=self.bin_count[v])
                self.BucketDict[f"Feature_{v}"] = self.Bucketer.fit(X[:, i])

        else:
            for i in range(X.shape[1]):
                self.Bucketer = QuantileBucketer(bin_count=self.bin_count)
                self.BucketDict[f"Feature_{i}"] = self.Bucketer.fit(X[:, i])

    def _transform(self, X, y=None):
        """Transforms a numerical array into its corresponding buckets using the fitted Quantile Probatus Bucketer.

        Args:
            X (np.array): The numerical data which will be transformed into the corresponding buckets

        Returns:
            np.array of the transformed X, such that the values of X are replaced by the corresponding bucket numbers
        """
        if isinstance(self.bin_count, dict):
            for i, v in enumerate(self.bin_count):
                X[:, i] = np.digitize(X[:, i], self.BucketDict[f"Feature_{v}"].boundaries[1:], right=True,)

        else:
            for i in range(X.shape[1]):
                X[:, i] = np.digitize(X[:, i], self.BucketDict[f"Feature_{i}"].boundaries[1:], right=True,)

        return X


class ManualBucketTransformer(BucketTransformer):
    """Bucket transformer implementing user-defined boundaries."""

    def __init__(self, mapping_config):
        """Initialise the user-defined boundaries with a YAML config file.

        Args:
            mapping_config (dict): Contains the feature names and boundaries defined for each feature
        """
        super().__init__()
        self.bin_count = mapping_config

    def _fit(self, X, y=None):
        """As the boundaries are already defined here, we do not need a fit function, and hence we keep this empty."""
        pass

    def _transform(self, X, y=None):
        """Transforms a numerical array into its corresponding buckets using the user-defined boundaries.

        Args:
            X (np.array): The numerical data which will be transformed into the corresponding buckets

        Returns:
            np.array of the transformed X, such that the values of X are replaced by the corresponding bucket numbers
        """
        for i, v in enumerate(self.bin_count):
            X[:, i] = np.digitize(X[:, i], self.bin_count[v]["boundaries"][1:], right=True,)
        return X
