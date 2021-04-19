import pandas as pd
import itertools

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from skorecard.reporting.plotting import PlotBucketMethod
from skorecard.reporting.report import BucketTableMethod


class BaseBucketer(BaseEstimator, TransformerMixin, PlotBucketMethod, BucketTableMethod):
    """Base class for bucket transformers."""

    @staticmethod
    def _is_dataframe(X):
        # checks if the input is a dataframe. Also creates a copy,
        # important not to transform the original dataset.
        if not isinstance(X, pd.DataFrame):
            raise TypeError("The data set should be a pandas dataframe")
        return X.copy()

    @staticmethod
    def _is_allowed_missing_treatment(missing_treatment):
        # checks if the argument for missing_values is valid
        allowed_str_missing = ["separate", "frequent", "risky"]
        if type(missing_treatment) == str:
            if missing_treatment not in allowed_str_missing:
                raise ValueError(f"missing_treatment must be in {allowed_str_missing} or a dict")

        elif type(missing_treatment) == dict:
            for _, v in enumerate(missing_treatment):
                if missing_treatment[v] < 0:
                    raise ValueError("As an integer, missing_treatment must be greater than 0")
                elif type(missing_treatment[v]) != int:
                    raise ValueError("Values of the missing_treatment dict must be integers")

        else:
            raise ValueError(f"missing_treatment must be in {allowed_str_missing} or a dict")

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
        flt = pd.isnull(X)
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

    def predict_proba(self, X):
        """Applies the transform method. To be used for the grid searches.

        Args:
            X (pd.DataFrame): The numerical data which will be transformed into the corresponding buckets

        Returns:
            np.array of the transformed X, such that the values of X are replaced by the corresponding bucket numbers
        """
        return self.transform(X)
