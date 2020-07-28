import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from probatus.binning import SimpleBucketer, AgglomerativeBucketer, QuantileBucketer


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
            raise AttributeError("Method not supported!")

        self.mapping = mapping  # todo: figure out mapping
        self.bin_count = bin_count
        self.method = method

    def fit(self, X):
        """Fits the relevant Probatus bucket onto the numerical array.

         We must generate a Bucket object per column in our dataset.

        Args:
            X (np.array): The numerical data on which we wish to fit our BucketTransformer
        """
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

                self.BucketDict[f"Bucketer_{self.method}_feature_{i}"] = self.Bucketer
                self.BucketDict[f"Bucketer_{self.method}_feature_{i}"].fit(X[:, i])

        else:
            # todo: apply mapping
            pass

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

    def fit_transform(self, X):
        """Apply in succession the fit and transform functions defined above.

        Args:
             X (np.array): The numerical data which will be transformed into the corresponding buckets

        Returns:
            np.array of the transformed X, such that the values of X are replaced by the corresponding bucket numbers
        """
        self.fit(X)
        return self.transform(X)
