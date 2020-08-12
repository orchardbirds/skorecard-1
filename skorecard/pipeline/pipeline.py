import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    """A slightly modified version of scikit lego's class.

    Found here:
    https://scikit-lego.readthedocs.io/en/latest/_modules/sklego/preprocessing/pandastransformers.html#ColumnSelector
    Allows selecting specific columns from a pandas DataFrame by name. Can be useful in a sklearn Pipeline.

    Args:
        columns: column name ``str`` or list of column names to be selected

    .. note::
        Raises a ``TypeError`` if input provided is not a DataFrame

        Raises a ``ValueError`` if columns provided are not in the input DataFrame

    """

    def __init__(self, columns: list):
        """Initialise ColumnSelector with columns, which must be a list."""
        self.columns = columns

    def fit(self, X, y=None):
        """Checks 1) if input is a DataFrame, and 2) if column names are in this DataFrame.

        Args:
            X: ``pd.DataFrame`` on which we apply the column selection
            y: ``pd.Series`` labels for X. unused for column selection

        Returns:
            ``ColumnSelector`` object.
        """
        self.columns_ = as_list(self.columns)
        self._check_X_for_type(X)
        self._check_column_length()
        self._check_column_names(X)
        return self

    def transform(self, X):
        """Returns a pandas DataFrame with only the specified columns.

        Args:
            X: ``pd.DataFrame`` on which we apply the column selection

        Returns:
            ``pd.DataFrame`` with only the selected columns
        """
        self._check_X_for_type(X)
        if self.columns:
            return X[self.columns_].values
        return X.values

    def get_feature_names(self):
        """Simply returns the columns."""
        return self.columns_

    def _check_column_length(self):
        """Check if no column is selected."""
        if len(self.columns_) == 0:
            raise ValueError("Expected columns to be at least of length 1, found length of 0 instead")

    def _check_column_names(self, X):
        """Check if one or more of the columns provided doesn't exist in the input DataFrame."""
        non_existent_columns = set(self.columns_).difference(X.columns)
        if len(non_existent_columns) > 0:
            raise KeyError(f"{list(non_existent_columns)} column(s) not in DataFrame")

    @staticmethod
    def _check_X_for_type(X):
        """Checks if input of the Selector is of the required dtype."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Provided variable X is not of type pandas.DataFrame")


def as_list(val):
    """Helper function, always returns a list of the input value, taken from scikit lego.

    Args:
        val: the input value.

    Returns:
        the input value as a list.
    """
    treat_single_value = str

    if isinstance(val, treat_single_value):
        return [val]

    if hasattr(val, "__iter__"):
        return list(val)

    return [val]
