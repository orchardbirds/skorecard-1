from skorecard.bucket_mapping import FeaturesBucketMapping
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


def get_features_bucket_mapping(pipe: Pipeline) -> FeaturesBucketMapping:
    """Get feature bucket mapping from a sklearn pipeline object.

    ```python
    from skorecard import datasets
    from skorecard.bucketers import EqualWidthBucketer, OrdinalCategoricalBucketer
    from skorecard.pipeline import get_features_bucket_mapping
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression

    X, y = datasets.load_uci_credit_card(return_X_y=True)

    bucket_pipeline = make_pipeline(
        EqualWidthBucketer(bins=5, variables=['LIMIT_BAL', 'BILL_AMT1']),
        OrdinalCategoricalBucketer(variables=['EDUCATION', 'MARRIAGE'])
    )

    pipe = Pipeline([
        ('bucketing', bucket_pipeline),
        ('one-hot-encoding', OneHotEncoder()),
        ('lr', LogisticRegression())
    ])
    
    pipe.fit(X, y)
    features_bucket_mapping = get_features_bucket_mapping(pipe)
    ```
    
    Args:
        pipe (Pipeline): fitted scikitlearn pipeline with bucketing transformers

    Returns:
        FeaturesBucketMapping: skorecard class with the bucket info
    """
    assert isinstance(pipe, Pipeline)

    features_bucket_mapping = {}
    for step in _get_all_steps(pipe):
        check_is_fitted(step)
        if hasattr(step, "features_bucket_mapping_"):
            features_bucket_mapping.update(step.features_bucket_mapping_)

    return FeaturesBucketMapping(features_bucket_mapping)


def _get_all_steps(pipeline):
    steps = []
    for named_step in pipeline.steps:
        step = named_step[1]
        if hasattr(step, "steps"):
            steps += _get_all_steps(step)
        else:
            steps.append(step)
    return steps


class ColumnSelector(BaseEstimator, TransformerMixin):
    """A slightly modified version of scikit lego's class.

    On the transform, we return X.values instead of X to be compatible with our Bucket Transformers

    The original can be found here:
    https://scikit-lego.readthedocs.io/en/latest/_modules/sklego/preprocessing/pandastransformers.html#ColumnSelector
    Allows selecting specific columns from a pandas DataFrame by name. Can be useful in a sklearn Pipeline.

    Args:
        columns (list): list of column names to be selected

    Note:
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
        self.columns_ = self._as_list(self.columns)
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

    @staticmethod
    def _as_list(val):
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
