import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from probatus.binning import SimpleBucketer, AgglomerativeBucketer, QuantileBucketer
import yaml


class BucketTransformer(BaseEstimator, TransformerMixin):
    """Buckets and transforms a numerical array using the Bucket methodologies in the Probatus package.

    Currently 3 methods are supported: Simple Bucketing, Agglomerative Bucketing and Quantile Bucketing

    """

    def __init__(self, bin_count=3, method="simple", mapping=None):
        """Initialise BucketTransformer using Probatus Bucket methodologies.

        Args:
            bin_count (int): How many bins we wish to split our data into. Required for each Probatus Bucket method
            method (str): "simple" - Creates equally spaced bins using numpy.histogram function
                          "agglomerative" - Bins by applying the Scikit-learn implementation of Agglomerative Clustering
                          "quantile" - Creates bins with equal number of elements
            mapping: Adds user-defined mapping. Not yet supported
        """
        if method not in [
            "simple",
            "agglomerative",
            "quantile",
        ]:  # todo: add more options?
            raise NotImplementedError("Method not supported!")

        self.mapping = mapping  # todo: figure out mapping
        self.bin_count = bin_count
        self.method = method

    def fit(self, X, y=None):
        """Fits the relevant Probatus bucket onto the numerical array.

         We must generate a Bucket object per column in our dataset.

        Args:
            X (np.array): The numerical data on which we wish to fit our BucketTransformer

        Returns:
            self (object): Fitted transformer
        """
        # 1-d arrays can cause us some hassle when looping over columns
        if X.ndim == 1:
            X = np.expand_dims(X, 1)

        if self.mapping is None:
            self.BucketDict = {}
            for i in range(X.shape[1]):
                if self.method == "simple":
                    self.Bucketer = SimpleBucketer(bin_count=self.bin_count)
                elif self.method == "agglomerative":
                    self.Bucketer = AgglomerativeBucketer(bin_count=self.bin_count)
                elif self.method == "quantile":
                    self.Bucketer = QuantileBucketer(bin_count=self.bin_count)

                self.BucketDict[f"Bucketer_{self.method}_feature_{i}"] = self.Bucketer.fit(X[:, i])

        else:
            # todo: apply mapping
            pass

        return self

    def transform(self, X):
        """Transforms a numerical array into its corresponding buckets using the fitted Probatus bucket methodology.

        Args:
            X (np.array): The numerical data which will be transformed into the corresponding buckets

        Returns:
            np.array of the transformed X, such that the values of X are replaced by the corresponding bucket numbers
        """
        X = X.copy()
        if X.ndim == 1:
            X = np.expand_dims(X, 1)

        # todo: apply mapping
        for i in range(X.shape[1]):
            X[:, i] = np.digitize(
                X[:, i], self.BucketDict[f"Bucketer_{self.method}_feature_{i}"].boundaries[1:], right=True,
            )
        return X

    def save(self, filename):
        """Save the self.BucketDict in a YAML file. A separate section is written per Bucket object in this dictionary.

        An example feature section is the following:

        Feature 0:
          bin_count: 5
          boundaries:
          - 10000.0
          - 160000.0
          - 310000.0
          - 460000.0
          - 610000.0
          - 760000.0
          method: simple

        Args:
            filename (str): Name of the YAML file that is saved
        """
        with open(f"{filename}.yaml", "w") as outfile:
            for i, v in enumerate(self.BucketDict):
                tmp_dict = {}
                tmp_dict["bin_count"] = self.BucketDict[v].bin_count
                tmp_dict["method"] = self.method
                tmp_dict["boundaries"] = self.BucketDict[v].boundaries.tolist()
                yaml.dump({f"Feature {i}": tmp_dict}, outfile, default_flow_style=False)
