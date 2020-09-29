import numpy as np
import pandas as pd
from probatus.binning import (
    SimpleBucketer,
    AgglomerativeBucketer,
    QuantileBucketer,
    TreeBucketer,
)

from .base_bucketer import BaseBucketer
from skorecard.bucket_mapping import BucketMapping, FeaturesBucketMapping


class EqualWidthBucketer(BaseBucketer):
    """Bucket transformer that creates equally spaced bins using numpy.histogram function.

    ```python
    from skorecard import datasets
    from skorecard.bucketers import EqualWidthBucketer

    X, y = datasets.load_uci_credit_card(return_X_y=True)
    bucketer = EqualWidthBucketer(bins = 10, variables = ['LIMIT_BAL'])
    bucketer.fit_transform(X)
    bucketer.fit_transform(X)['LIMIT_BAL'].value_counts()
    ```
    """

    def __init__(self, bins=-1, variables=[]):
        """Init the class.

        Args:
            bins (int): Number of bins to create.
            variables (list): The features to bucket. Uses all features if not defined.
        """
        assert isinstance(variables, list)
        assert isinstance(bins, int)

        self.variables = variables
        self.bins = bins

        self.bucketer = SimpleBucketer(bin_count=self.bins)

    def fit(self, X, y=None):
        """Fit X, y."""
        super().fit(X, y=None)  # Set y to None, as this bucketer doesn't use it
        return self

    def transform(self, X):
        """Transform X."""
        return super().transform(X)


class AgglomerativeClusteringBucketer(BaseBucketer):
    """Bucket transformer that creates bins using sklearn.AgglomerativeClustering.

    ```python
    from skorecard import datasets
    from skorecard.bucketers import AgglomerativeClusteringBucketer

    X, y = datasets.load_uci_credit_card(return_X_y=True)
    bucketer = AgglomerativeClusteringBucketer(bins = 10, variables=['LIMIT_BAL'])
    bucketer.fit_transform(X)
    bucketer.fit_transform(X)['LIMIT_BAL'].value_counts()
    ```
    """

    def __init__(self, bins=-1, variables=[]):
        """Init the class.

        Args:
            bins (int): Number of bins to create.
            variables (list): The features to bucket. Uses all features if not defined.
        """
        assert isinstance(variables, list)
        assert isinstance(bins, int)

        self.variables = variables
        self.bins = bins

        self.bucketer = AgglomerativeBucketer(bin_count=self.bins)

    def fit(self, X, y=None):
        """Fit X, y."""
        super().fit(X, y=None)  # Set y to None, as this bucketer doesn't use it
        return self

    def transform(self, X):
        """Transform X."""
        return super().transform(X)


class EqualFrequencyBucketer(BaseBucketer):
    """Bucket transformer that creates bins with equal number of elements.

    ```python
    from skorecard import datasets
    from skorecard.bucketers import EqualFrequencyBucketer

    X, y = datasets.load_uci_credit_card(return_X_y=True)
    bucketer = EqualFrequencyBucketer(bins = 10, variables=['LIMIT_BAL'])
    bucketer.fit_transform(X)
    bucketer.fit_transform(X)['LIMIT_BAL'].value_counts()
    ```
    """

    def __init__(self, bins=-1, variables=[]):
        """Init the class.

        Args:
            bins (int): Number of bins to create.
            variables (list): The features to bucket. Uses all features if not defined.
        """
        assert isinstance(variables, list)
        assert isinstance(bins, int)

        self.variables = variables
        self.bins = bins

        self.bucketer = QuantileBucketer(bin_count=self.bins)

    def fit(self, X, y=None):
        """Fit X, y."""
        super().fit(X, y=None)  # Set y to None, as this bucketer doesn't use it
        return self

    def transform(self, X):
        """Transform X."""
        return super().transform(X)


class DecisionTreeBucketer(BaseBucketer):
    """Bucket transformer that creates bins by training a decision tree.

    ```python
    from skorecard import datasets
    from skorecard.bucketers import DecisionTreeBucketer

    X, y = datasets.load_uci_credit_card(return_X_y=True)
    dt_bucketer = DecisionTreeBucketer(variables=['LIMIT_BAL'])
    dt_bucketer.fit_transform(X, y)
    dt_bucketer.fit_transform(X, y)['LIMIT_BAL'].value_counts()
    ```
    """

    def __init__(self, variables=[], **kwargs):
        """Init the class.

        Args:
            variables (list): The features to bucket. Uses all features if not defined.
        """
        assert isinstance(variables, list)

        self.variables = variables
        self.kwargs = kwargs

        self.bucketer = TreeBucketer(**self.kwargs)

    def fit(self, X, y):
        """Fit X,y."""
        super().fit(X, y)
        return self

    def transform(self, X):
        """Transform X."""
        return super().transform(X)

    def get_params(self, deep=True):
        """Return the parameters of the decision tree used in the Transformer.

        Args:
            deep (bool): Make a deep copy or not, required by the API.

        Returns:
            (dict): Decision Tree Parameters
        """
        return self.bucketer.tree.get_params(deep=deep)

    def set_params(self, **params):
        """Set the parameteres for the decision tree.

        Args:
            **params: (dict) parameters for the decision tree

        """
        self.bucketer.tree.set_params(**params)
        return self


class OrdinalCategoricalBucketer(BaseBucketer):
    """
    The OrdinalCategoricalEncoder() replaces categories by ordinal numbers.

    Example (0, 1, 2, 3, etc). The numbers are assigned ordered based on the mean of the target
    per category, or assigned in order of frequency when y is not.

    Ordered ordinal encoding: for the variable colour, if the mean of the target
    for blue, red and grey is 0.5, 0.8 and 0.1 respectively, blue is replaced by 2,
    red by 3 and grey by 1. If new data contains unknown labels (f.e. yellow),
    they will be replaced by 0.

    Credits: Code & ideas adapted from
    - feature_engine.categorical_encoders.OrdinalCategoricalEncoder
    - feature_engine.categorical_encoders.RareLabelCategoricalEncoder

    ```python
    from skorecard import datasets
    from skorecard.bucketers import OrdinalCategoricalBucketer

    X, y = datasets.load_uci_credit_card(return_X_y=True)
    bucketer = OrdinalCategoricalBucketer(variables=['EDUCATION'])
    bucketer.fit_transform(X)
    bucketer = OrdinalCategoricalBucketer(max_n_categories=2, variables=['EDUCATION'])
    bucketer.fit_transform(X, y)
    ```
    """

    def __init__(self, tol=0.05, max_n_categories=None, variables=[]):
        """Init the class.

        Args:
            tol (float): the minimum frequency a label should have to be considered frequent.
                Categories with frequencies lower than tol will be grouped together (in the highest ).
            max_n_categories (int): the maximum number of categories that should be considered frequent.
                If None, all categories with frequency above the tolerance (tol) will be
                considered.
            variables (list): The features to bucket. Uses all features if not defined.
        """
        assert isinstance(variables, list)

        if tol < 0 or tol > 1:
            raise ValueError("tol takes values between 0 and 1")

        if max_n_categories is not None:
            if max_n_categories < 0 or not isinstance(max_n_categories, int):
                raise ValueError("max_n_categories takes only positive integer numbers")

        self.tol = tol
        self.max_n_categories = max_n_categories
        self.variables = variables

    def fit(self, X, y=None):
        """Init the class."""
        X = self._is_dataframe(X)
        self.variables = self._check_variables(X, self.variables)

        self.features_bucket_mapping_ = {}

        for var in self.variables:

            # Normalized counts
            normalized_counts = X[var].value_counts() / X.shape[0]

            # Determine the order of unique values
            if y is not None:
                X["target"] = y
                cats = X.groupby([var])["target"].mean().sort_values(ascending=True).index
                normalized_counts = normalized_counts[cats]

            # Limit number of categories if set.
            normalized_counts = normalized_counts[: self.max_n_categories]

            # Remove less frequent categories
            normalized_counts = normalized_counts[normalized_counts >= self.tol]

            # Determine Ordinal Encoder based on ordered labels
            # Note we start at 1, to be able to encode missings as 0.
            mapping = dict(zip(normalized_counts.index, range(1, len(normalized_counts) + 1)))

            self.features_bucket_mapping_[var] = BucketMapping(
                feature_name=var, type="categorical", map=mapping, missing_bucket=0
            )

        return self

    def transform(self, X):
        """Transform X."""
        return super().transform(X)


class UserInputBucketer(BaseBucketer):
    """Bucket transformer implementing user-defined boundaries.

    ```python
    from skorecard import datasets
    from skorecard.bucketers import AgglomerativeClusteringBucketer, UserInputBucketer

    X, y = datasets.load_uci_credit_card(return_X_y=True)

    ac_bucketer = AgglomerativeClusteringBucketer(bins=3, variables=['LIMIT_BAL'])
    ac_bucketer.fit(X)
    mapping = ac_bucketer.features_bucket_mapping_

    ui_bucketer = UserInputBucketer(mapping)
    new_X = ui_bucketer.fit_transform(X)
    assert len(new_X['LIMIT_BAL'].unique()) == 3
    ```
    """

    def __init__(self, features_bucket_mapping, variables=[]):
        """Initialise the user-defined boundaries with a dictionary.

        Args:
            features_bucket_mapping (dict): Contains the feature name and boundaries defined for this feature.
                Either dict or FeaturesBucketMapping
            variables (list): The features to bucket. Uses all features if not defined.
        """
        # Check user defined input for bucketing. If a dict is specified, will auto convert
        if not isinstance(features_bucket_mapping, FeaturesBucketMapping):
            if not isinstance(features_bucket_mapping, dict):
                raise TypeError("'features_bucket_mapping' must be a dict or FeaturesBucketMapping instance")
            features_bucket_mapping = FeaturesBucketMapping(features_bucket_mapping)

        self.features_bucket_mapping_ = features_bucket_mapping

        # If user did not specify any variables,
        # use all the variables defined in the features_bucket_mapping
        self.variables = variables
        if variables == []:
            self.variables = list(self.features_bucket_mapping_.maps.keys())

    def fit(self, X, y=None):
        """Init the class."""
        return self

    def transform(self, X):
        """Transform X."""
        return super().transform(X)


class WoEBucketer(BaseBucketer):
    """Transformer to map the bucketed features to their Weight of Evidence estimation.

    Only works for binary classification.

    The weight of evidence is given by: `np.log( p(1) / p(0) )`
    The target probability ratio is given by: `p(1) / p(0)`

    For example in the variable colour, if the mean of the target = 1 for blue is 0.8 and
    the mean of the target = 0 is 0.2, blue will be replaced by: np.log(0.8/0.2) = 1.386
    if log_ratio is selected. Alternatively, blue will be replaced by 0.8 / 0.2 = 4 if ratio is selected.

    For details on the weight of evidence:
    https://multithreaded.stitchfix.com/blog/2015/08/13/weight-of-evidence/

    ```python
    from skorecard import datasets
    from skorecard.bucketers import WoEBucketer

    X, y = datasets.load_uci_credit_card(return_X_y=True)
    woeb = WoEBucketer(variables=['EDUCATION'])
    woeb.fit_transform(X, y)
    woeb.fit_transform(X, y)['EDUCATION'].value_counts()
    ```

    Credit: Some code taken from feature_engine.categorical_encoders.

    """

    def __init__(self, epsilon=0.0001, variables=[]):
        """Constructor for WoEBucketer.

        Args:
            epsilon (float): Amount to be added to relative counts in order to avoid division by zero in the WOE
            calculation.
            variables (list): The features to bucket. Uses all features if not defined.
        """
        assert isinstance(variables, list)
        assert epsilon >= 0

        self.epsilon = epsilon
        self.variables = variables

    def fit(self, X, y):
        """Calculate the WOE for every column.

        Args:
            X (np.array): (binned) features
            y (np.array): target
        """
        assert y is not None, "WoEBucketer needs a target y"
        assert len(np.unique(y)) == 2, "WoEBucketer is only suited for binary classification"

        X = self._is_dataframe(X)

        # TODO: WoE should treat missing values as a separate bin and thus handled seamlessly.
        self._check_contains_na(X, self.variables)

        X["target"] = y
        self.features_bucket_mapping_ = {}

        total_pos = X["target"].sum()
        total_neg = X.shape[0] - total_pos
        X["non_target"] = np.where(X["target"] == 1, 0, 1)

        for var in self.variables:
            pos = X.groupby([var])["target"].sum() / total_pos
            neg = X.groupby([var])["non_target"].sum() / total_neg

            t = pd.concat([pos + self.epsilon, neg + self.epsilon], axis=1)
            t["woe"] = np.log(t["target"] / t["non_target"])

            if any(t["target"] == 0) or any(t["non_target"] == 0):
                raise ZeroDivisionError("One of the categories has no occurances of the positive class")

            self.features_bucket_mapping_[var] = BucketMapping(
                feature_name=var, type="categorical", map=t["woe"].to_dict()
            )

        return self

    def transform(self, X):
        """Transform X."""
        return super().transform(X)
