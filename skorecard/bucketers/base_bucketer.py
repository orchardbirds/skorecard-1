import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from skorecard.bucket_mapping import BucketMapping
from sklearn.utils.validation import check_is_fitted


class BaseBucketer(BaseEstimator, TransformerMixin):
    """Base class for bucket transformers."""

    @staticmethod
    def _is_dataframe(X):
        # checks if the input is a dataframe. Also creates a copy,
        # important not to transform the original dataset.
        if not isinstance(X, pd.DataFrame):
            raise TypeError("The data set should be a pandas dataframe")
        return X.copy()

    @staticmethod
    def _check_contains_na(X, variables):

        has_missings = X[variables].isnull().any()
        vars_missing = has_missings[has_missings].index.tolist()

        if vars_missing:
            raise ValueError(f"The variables {vars_missing} contain missing values. Consider using an imputer first.")

    @staticmethod
    def _check_variables(X, variables):
        assert isinstance(variables, list)
        if len(variables) == 0:
            return X.columns
        else:
            for var in variables:
                assert var in X.columns, f"Column {var} not present in X"
            return variables

    def fit(self, X, y=None):
        """Fit X, y."""
        X = self._is_dataframe(X)
        self.variables = self._check_variables(X, self.variables)

        self.features_bucket_mapping_ = {}

        for feature in self.variables:
            self.bucketer.fit(X[feature].values, y)
            self.features_bucket_mapping_[feature] = BucketMapping(
                feature_name=feature, type="numerical", map=self.bucketer.boundaries, missing_bucket=None,
            )

        return self

    def transform(self, X, y=None):
        """Transforms an array into the corresponding buckets fitted by the Transformer.

        Args:
            X (pd.DataFrame): dataframe which will be transformed into the corresponding buckets
            y (array): target

        Returns:
            pd.DataFrame with transformed features
        """
        check_is_fitted(self)
        X = self._is_dataframe(X)

        for feature in self.variables:
            bucket_mapping = self.features_bucket_mapping_.get(feature)
            X[feature] = bucket_mapping.transform(X[feature])

        return X

    def predict(self, X, y=None):
        """Applies the transform method. To be used for the grid searches.

        Args:
            X (pd.DataFrame): The numerical data which will be transformed into the corresponding buckets

        Returns:
            np.array of the transformed X, such that the values of X are replaced by the corresponding bucket numbers
        """
        return self.transform(X, y)


# def _check_contains_na(X, variables):
#     if X[variables].isnull().values.any():
#         raise ValueError('Some of the variables to transform contain missing values. Check and remove those '
#                          'before using this transformer.')


# df = pd.DataFrame(['a','b','a'],columns=['some_col'])
# from feature_engine.categorical_encoders import OrdinalCategoricalEncoder
# OrdinalCategoricalEncoder(encoding_method='arbitrary').fit_transform(df)
