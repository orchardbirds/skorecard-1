import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils.validation import check_is_fitted
import itertools


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
            variables = list(X.columns)
        else:
            for var in variables:
                assert var in list(X.columns), f"Column {var} not present in X"
        assert variables is not None and len(variables) > 0
        return variables

    @staticmethod
    def _filter_specials_for_fit(X, y, specials):
        """
        We need to filter out the specials from a vector.

        Because we don't want to use those values to determine bin boundaries.
        """
        flt_vals = list(itertools.chain(*specials.values()))
        flt = X.isin(flt_vals)
        X_out = X[~flt]
        if y is not None:
            y_out = y[~flt]
        else:
            y_out = y
        return X_out, y_out

    def _filter_na_for_fit(self, X, y):
        """
        We need to filter out the missing values from a vector.

        Because we don't want to use those values to determine bin boundaries.
        """
        flt = np.isnan(X)
        X_out = X[~flt]
        if y is not None:
            y_out = y[~flt]
        else:
            y_out = y
        return X_out, y_out

    @staticmethod
    def _verify_specials_variables(specials, variables):
        """
        Make sure all specials columns are also in the data.
        """
        diff = set(specials.keys()).difference(set(variables))
        if len(diff) > 0:
            raise ValueError(f"Features {diff} are defined in the specials dictionary, but not in the variables.")

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

    def predict(self, X):
        """Applies the transform method. To be used for the grid searches.

        Args:
            X (pd.DataFrame): The numerical data which will be transformed into the corresponding buckets

        Returns:
            np.array of the transformed X, such that the values of X are replaced by the corresponding bucket numbers
        """
        return self.transform(X)


# def _check_contains_na(X, variables):
#     if X[variables].isnull().values.any():
#         raise ValueError('Some of the variables to transform contain missing values. Check and remove those '
#                          'before using this transformer.')


# df = pd.DataFrame(['a','b','a'],columns=['some_col'])
# from feature_engine.categorical_encoders import OrdinalCategoricalEncoder
# OrdinalCategoricalEncoder(encoding_method='arbitrary').fit_transform(df)
