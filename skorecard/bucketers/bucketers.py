import warnings
import numpy as np
import pandas as pd
from probatus.binning import AgglomerativeBucketer
from probatus.utils import ApproximationWarning

from typing import Union, List, Dict
from skorecard.bucketers.base_bucketer import BaseBucketer
from skorecard.bucket_mapping import BucketMapping, FeaturesBucketMapping
from skorecard.utils import NotInstalledError, NotPreBucketedError
from skorecard.reporting import build_bucket_table

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree

try:
    from optbinning import OptimalBinning
except ModuleNotFoundError:
    OptimalBinning = NotInstalledError("optbinning")


class OptimalBucketer(BaseBucketer):
    """Find Optimal Buckets.

    Bucket transformer that uses the [optbinning](http://gnpalencia.org/optbinning) package to find optimal buckets.
    This bucketers basically wraps optbinning.OptimalBinning to be consistent with skorecard.

    This bucketer uses pre-binning to bucket a feature into max 100 bins. It then uses a constrained programming solver
    to merge buckets, taking into accounts constraints 1) monotonicity in bad rate, 2) at least 5% of records per bin.

    This bucketer:

    - Is supervised: is uses the target variable to find good buckets
    - Supports both numerical and categorical features

    Example:

    ```python
    from skorecard import datasets
    from skorecard.bucketers import OptimalBucketer

    X, y = datasets.load_uci_credit_card(return_X_y=True)
    bucketer = OptimalBucketer(variables = ['LIMIT_BAL'])
    bucketer.fit_transform(X, y)
    ```
    """

    def __init__(
        self,
        variables=[],
        specials={},
        variables_type="numerical",
        max_n_bins=10,
        missing_treatment="separate",
        min_bin_size=0.05,
        cat_cutoff=None,
        time_limit=25,
        **kwargs,
    ) -> None:
        """Initialize Optimal Bucketer.

        Args:
            variables: List of variables to bucket.
            specials: (nested) dictionary of special values that require their own binning.
                The dictionary has the following format:
                 {"<column name>" : {"name of special bucket" : <list with 1 or more values>}}
                For every feature that needs a special value, a dictionary must be passed as value.
                This dictionary contains a name of a bucket (key) and an array of unique values that should be put
                in that bucket.
                When special values are passed, they are not considered in the fitting procedure.
            variables_type: Type of the variables
            missing_treatment: Defines how we treat the missing values present in the data.
                If a string, it must be in ['separate', 'risky', 'frequent']
                separate: Missing values get put in their separate bucket
                risky: todo
                frequent: todo

                If a dict, it must be of the following format:
                {"<column name>": <bucket_number>}
                This bucket number is where we will put the missing values.
            min_bin_size: Minimum fraction of observations in a bucket. Passed to optbinning.OptimalBinning.
            max_n_bins: Maximum numbers of bins to return. Passed to optbinning.OptimalBinning.
            cat_cutoff: Threshold ratio (None, or >0 and <=1) below which categories are grouped
                together in a bucket 'other'. Passed to optbinning.OptimalBinning.
            time_limit: Time limit in seconds to find an optimal solution. Passed to optbinning.OptimalBinning.
            kwargs: Other parameters passed to optbinning.OptimalBinning. Passed to optbinning.OptimalBinning.
        """
        self._is_allowed_missing_treatment(missing_treatment)
        assert variables_type in ["numerical", "categorical"]

        self.variables = variables
        self.specials = specials
        self.variables_type = variables_type
        self.max_n_bins = max_n_bins
        self.missing_treatment = missing_treatment
        self.min_bin_size = min_bin_size
        self.cat_cutoff = cat_cutoff
        self.time_limit = time_limit

        self.kwargs = kwargs

        # not tested right now

    def fit(self, X, y):
        """Fit X, y."""
        X = self._is_dataframe(X)
        self.variables = self._check_variables(X, self.variables)
        self._verify_specials_variables(self.specials, X.columns)

        if isinstance(y, pd.Series):
            y = y.values

        self.features_bucket_mapping_ = {}
        self.bucket_tables_ = {}
        self.binners = {}

        for feature in self.variables:

            if feature in self.specials.keys():
                special = self.specials[feature]
                X_flt, y_flt = self._filter_specials_for_fit(X=X[feature], y=y, specials=special)
            else:
                X_flt, y_flt = X[feature], y
                special = {}
            if self.variables_type == "numerical":
                X_flt, y_flt = self._filter_na_for_fit(X=X_flt, y=y_flt)
                uniq_values = np.sort(np.unique(X_flt.values))
                if len(uniq_values) > 100:
                    raise NotPreBucketedError(
                        f"""
                        OptimalBucketer requires numerical feature '{feature}' to be pre-bucketed
                        to max 100 unique values (for performance reasons).
                        Currently there are {len(uniq_values)} unique values present.

                        Apply pre-binning, f.e. with skorecard.bucketers.DecisionTreeBucketer.
                        """
                    )
                user_splits = uniq_values
            else:
                X_flt, y_flt = self._filter_na_for_fit(X=X_flt, y=y_flt)
                user_splits = None

            binner = OptimalBinning(
                name=feature,
                dtype=self.variables_type,
                solver="cp",
                monotonic_trend="auto_asc_desc",
                # We want skorecard users to explicitly define pre-binning for numerical features
                # Setting the user_splits prevents OptimalBinning from doing pre-binning again.
                user_splits=user_splits,
                min_bin_size=self.min_bin_size,
                max_n_bins=self.max_n_bins,
                cat_cutoff=self.cat_cutoff,
                time_limit=self.time_limit,
                **self.kwargs,
            )
            self.binners[feature] = binner

            binner.fit(X_flt.values, y_flt)

            # Extract fitted boundaries
            if self.variables_type == "categorical":
                splits = {}
                for bucket_nr, values in enumerate(binner.splits):
                    for value in values:
                        splits[value] = bucket_nr
            else:
                splits = binner.splits

            # Note that optbinning transform uses right=False
            # https://github.com/guillermo-navas-palencia/optbinning/blob/396b9bed97581094167c9eb4744c2fd1fb5c7408/optbinning/binning/transformations.py#L126-L132
            self.features_bucket_mapping_[feature] = BucketMapping(
                feature_name=feature,
                type=self.variables_type,
                map=splits,
                right=False,
                specials=special,
                missing_treatment=self.missing_treatment,
            )

        return self

    def transform(self, X):
        """Transform X."""
        return super().transform(X)


class EqualWidthBucketer(BaseBucketer):
    """Bucket transformer that creates equally spaced bins using numpy.histogram function.

    This bucketer:
    - is unsupervised: it does not consider the target value when fitting the buckets.
    - ignores missing values and passes them through.

    ```python
    from skorecard import datasets
    from skorecard.bucketers import EqualWidthBucketer

    specials = {"LIMIT_BAL": {"=50000": [50000], "in [20001,30000]": [20000, 30000]}}

    X, y = datasets.load_uci_credit_card(return_X_y=True)
    bucketer = EqualWidthBucketer(n_bins = 10, variables = ['LIMIT_BAL'], specials=specials)
    bucketer.fit_transform(X)
    bucketer.fit_transform(X)['LIMIT_BAL'].value_counts()
    ```
    """

    def __init__(self, n_bins=-1, variables=[], specials={}, missing_treatment="separate"):
        """Init the class.

        Args:
            n_bins (int): Number of bins to create.
            variables (list): The features to bucket. Uses all features if not defined.
            specials: (dict) of special values that require their own binning.
                The dictionary has the following format:
                 {"<column name>" : {"name of special bucket" : <list with 1 or more values>}}
                For every feature that needs a special value, a dictionary must be passed as value.
                This dictionary contains a name of a bucket (key) and an array of unique values that should be put
                in that bucket.
                When special values are defined, they are not considered in the fitting procedure.
            missing_treatment: Defines how we treat the missing values present in the data.
                If a string, it must be in ['separate', 'risky', 'frequent']
                separate: Missing values get put in their separate bucket
                risky: todo
                frequent: todo

                If a dict, it must be of the following format:
                {"<column name>": <bucket_number>}
                This bucket number is where we will put the missing values.
        """
        assert isinstance(variables, list)
        assert isinstance(n_bins, int)
        self._is_allowed_missing_treatment(missing_treatment)

        self.missing_treatment = missing_treatment
        self.variables = variables
        self.n_bins = n_bins
        self.specials = specials

    def fit(self, X, y=None):
        """Fit X, y."""
        X = self._is_dataframe(X)
        self.variables = self._check_variables(X, self.variables)
        self._verify_specials_variables(self.specials, X.columns)

        self.features_bucket_mapping_ = {}
        self.bucket_tables_ = {}

        for feature in self.variables:

            if feature in self.specials.keys():
                special = self.specials[feature]
                X_flt, y_flt = self._filter_specials_for_fit(X=X[feature], y=y, specials=special)
            else:
                X_flt = X[feature]
                y_flt = y
                special = {}

            X_flt, y_flt = self._filter_na_for_fit(X=X_flt, y=y_flt)
            _, boundaries = np.histogram(X_flt.values, bins=self.n_bins)

            # np.histogram returns the min & max values of the fits
            # On transform, we use np.digitize, which means new data that is outside of this range
            # will be assigned to their own buckets.
            # To solve, we simply remove the min and max boundaries
            boundaries = boundaries[1:-1]

            self.features_bucket_mapping_[feature] = BucketMapping(
                feature_name=feature,
                type="numerical",
                map=boundaries,
                right=True,
                specials=special,
                missing_treatment=self.missing_treatment,
            )

            # Calculate the bucket table
            self.bucket_tables_[feature] = build_bucket_table(
                X, y, column=feature, bucket_mapping=self.features_bucket_mapping_.get(feature)
            )

        return self


class AgglomerativeClusteringBucketer(BaseBucketer):
    """Bucket transformer that creates bins using sklearn.AgglomerativeClustering.

    This bucketer:
    - is unsupervised: it does not consider the target value when fitting the buckets.
    - ignores missing values and passes them through.

    ```python
    from skorecard import datasets
    from skorecard.bucketers import AgglomerativeClusteringBucketer

    specials = {"LIMIT_BAL": {"=50000": [50000], "in [20001,30000]": [20000, 30000]}}

    X, y = datasets.load_uci_credit_card(return_X_y=True)
    bucketer = AgglomerativeClusteringBucketer(n_bins = 10, variables=['LIMIT_BAL'], specials=specials)
    bucketer.fit_transform(X)
    bucketer.fit_transform(X)['LIMIT_BAL'].value_counts()
    ```
    """

    def __init__(self, n_bins=-1, variables=[], specials={}, missing_treatment="separate"):
        """Init the class.

        Args:
            n_bins (int): Number of bins to create.
            variables (list): The features to bucket. Uses all features if not defined.
            specials: (dict) of special values that require their own binning.
                The dictionary has the following format:
                 {"<column name>" : {"name of special bucket" : <list with 1 or more values>}}
                For every feature that needs a special value, a dictionary must be passed as value.
                This dictionary contains a name of a bucket (key) and an array of unique values that should be put
                in that bucket.
                When special values are defined, they are not considered in the fitting procedure.
            missing_treatment: Defines how we treat the missing values present in the data.
                If a string, it must be in ['separate', 'risky', 'frequent']
                separate: Missing values get put in their separate bucket
                risky: todo
                frequent: todo

                If a dict, it must be of the following format:
                {"<column name>": <bucket_number>}
                This bucket number is where we will put the missing values.

        """
        assert isinstance(variables, list)
        assert isinstance(n_bins, int)
        self._is_allowed_missing_treatment(missing_treatment)

        self.variables = variables
        self.n_bins = n_bins
        self.specials = specials
        self.missing_treatment = missing_treatment

    def fit(self, X, y=None):
        """Fit X, y."""
        X = self._is_dataframe(X)
        self.variables = self._check_variables(X, self.variables)
        self._verify_specials_variables(self.specials, X.columns)

        self.features_bucket_mapping_ = {}
        self.bucket_tables_ = {}

        for feature in self.variables:
            ab = AgglomerativeBucketer(bin_count=self.n_bins)

            if feature in self.specials.keys():
                special = self.specials[feature]
                X_flt, y_flt = self._filter_specials_for_fit(X=X[feature], y=y, specials=special)
            else:
                X_flt = X[feature]
                y_flt = y
                special = {}
            X_flt, y_flt = self._filter_na_for_fit(X=X_flt, y=y_flt)
            ab.fit(X_flt.values, y=None)

            # AgglomerativeBucketer returns the min & max values of the fits
            # On transform, we use np.digitize, which means new data that is outside of this range
            # will be assigned to their own buckets.
            # To solve, we remove the min and max boundaries
            boundaries = ab.boundaries[1:-1]

            self.features_bucket_mapping_[feature] = BucketMapping(
                feature_name=feature,
                type="numerical",
                map=boundaries,
                right=True,
                specials=special,
                missing_treatment=self.missing_treatment,
            )

            # Calculate the bucket table
            self.bucket_tables_[feature] = build_bucket_table(
                X, y, column=feature, bucket_mapping=self.features_bucket_mapping_.get(feature)
            )

        return self


class EqualFrequencyBucketer(BaseBucketer):
    """Bucket transformer that creates bins with equal number of elements.

    This bucketer:
    - is unsupervised: it does not consider the target value when fitting the buckets.
    - ignores missing values and passes them through.

    ```python
    from skorecard import datasets
    from skorecard.bucketers import EqualFrequencyBucketer

    X, y = datasets.load_uci_credit_card(return_X_y=True)
    bucketer = EqualFrequencyBucketer(n_bins = 10, variables=['LIMIT_BAL'])
    bucketer.fit_transform(X)
    bucketer.fit_transform(X)['LIMIT_BAL'].value_counts()
    ```
    """

    def __init__(self, n_bins=-1, variables=[], specials={}, missing_treatment="separate"):
        """Init the class.

        Args:
            n_bins (int): Number of bins to create.
            variables (list): The features to bucket. Uses all features if not defined.
            specials: (nested) dictionary of special values that require their own binning.
                The dictionary has the following format:
                 {"<column name>" : {"name of special bucket" : <list with 1 or more values>}}
                For every feature that needs a special value, a dictionary must be passed as value.
                This dictionary contains a name of a bucket (key) and an array of unique values that should be put
                in that bucket.
                When special values are defined, they are not considered in the fitting procedure.
            missing_treatment: Defines how we treat the missing values present in the data.
                If a string, it must be in ['separate', 'risky', 'frequent']
                separate: Missing values get put in their separate bucket
                risky: todo
                frequent: todo

                If a dict, it must be of the following format:
                {"<column name>": <bucket_number>}
                This bucket number is where we will put the missing values.

        """
        assert isinstance(variables, list)
        assert isinstance(n_bins, int)
        self._is_allowed_missing_treatment(missing_treatment)

        self.variables = variables
        self.n_bins = n_bins
        self.specials = specials
        self.missing_treatment = missing_treatment

    def fit(self, X, y=None):
        """Fit X, y.

        Uses pd.qcut()
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.qcut.html

        """
        X = self._is_dataframe(X)
        self.variables = self._check_variables(X, self.variables)
        self._verify_specials_variables(self.specials, X.columns)

        self.features_bucket_mapping_ = {}
        self.bucket_tables_ = {}

        for feature in self.variables:

            if feature in self.specials.keys():
                special = self.specials[feature]
                X_flt, y_flt = self._filter_specials_for_fit(X=X[feature], y=y, specials=special)
            else:
                X_flt = X[feature]
                special = {}
            try:
                _, boundaries = pd.qcut(X_flt, q=self.n_bins, retbins=True, duplicates="raise")
            except ValueError:
                # If there are too many duplicate values (assume a lot of filled missings)
                # this crashes - the exception drops them.
                # This means that it will return approximate quantile bins
                _, boundaries = pd.qcut(X_flt, q=self.n_bins, retbins=True, duplicates="drop")
                warnings.warn(ApproximationWarning("Approximated quantiles - too many unique values"))

            # pd.qcut returns the min & max values of the fits
            # On transform, we use np.digitize, which means new data that is outside of this range
            # will be assigned to their own buckets.
            # To solve, we simply remove the min and max boundaries
            boundaries = boundaries[1:-1]

            self.features_bucket_mapping_[feature] = BucketMapping(
                feature_name=feature,
                type="numerical",
                map=boundaries,
                right=True,  # pd.qcut returns bins including right edge: (edge, edge]
                specials=special,
                missing_treatment=self.missing_treatment,
            )

            # Calculate the bucket table
            self.bucket_tables_[feature] = build_bucket_table(
                X, y, column=feature, bucket_mapping=self.features_bucket_mapping_.get(feature)
            )

        return self


class DecisionTreeBucketer(BaseBucketer):
    """Bucket transformer that creates bins by training a decision tree.

    This bucketer:
    - is supervised: it uses the target value when fitting the buckets.
    - ignores missing values and passes them through.

    It uses
    [sklearn.tree.DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
    to find the splits.

    ```python
    from skorecard import datasets
    from skorecard.bucketers import DecisionTreeBucketer
    X, y = datasets.load_uci_credit_card(return_X_y=True)

    # make sure that those cases
    specials = {
        "LIMIT_BAL":{
            "=50000":[50000],
            "in [20001,30000]":[20000,30000],
            }
    }

    dt_bucketer = DecisionTreeBucketer(variables=['LIMIT_BAL'], specials = specials)
    dt_bucketer.fit(X, y)

    dt_bucketer.fit_transform(X, y)['LIMIT_BAL'].value_counts()
    ```
    """

    def __init__(
        self,
        variables=[],
        specials={},
        max_n_bins=100,
        missing_treatment="separate",
        min_bin_size=0.05,
        random_state=42,
        **kwargs,
    ) -> None:
        """Init the class.

        Args:
            variables (list): The features to bucket. Uses all features if not defined.
            specials (dict):  dictionary of special values that require their own binning.
                The dictionary has the following format:
                 {"<column name>" : {"name of special bucket" : <list with 1 or more values>}}
                For every feature that needs a special value, a dictionary must be passed as value.
                This dictionary contains a name of a bucket (key) and an array of unique values that should be put
                in that bucket.
                When special values are defined, they are not considered in the fitting procedure.
            min_bin_size: Minimum fraction of observations in a bucket. Passed directly to min_samples_leaf.
            max_n_bins: Maximum numbers of after the bucketing. Passed directly to max_leaf_nodes of the
                DecisionTreeClassifier.
                If specials are defined, max_leaf_nodes will be redefined to max_n_bins - (number of special bins).
                The DecisionTreeClassifier requires max_leaf_nodes>=2:
                therefore, max_n_bins  must always be >= (number of special bins + 2) if specials are defined,
                otherwise must be >=2.
            missing_treatment: Defines how we treat the missing values present in the data.
                If a string, it must be in ['separate', 'risky', 'frequent']
                separate: Missing values get put in their separate bucket
                risky: todo
                frequent: todo

                If a dict, it must be of the following format:
                {"<column name>": <bucket_number>}
                This bucket number is where we will put the missing values.
            random_state: The random state, Passed directly to DecisionTreeClassifier
            kwargs: Other parameters passed to DecisionTreeClassifier
        """
        assert isinstance(variables, list)
        self._is_allowed_missing_treatment(missing_treatment)

        self.variables = variables
        self.specials = specials
        self.kwargs = kwargs
        self.max_n_bins = max_n_bins
        self.missing_treatment = missing_treatment
        self.min_bin_size = min_bin_size
        self.random_state = random_state

    def fit(self, X, y):
        """Fit X, y."""
        X = self._is_dataframe(X)
        self.variables = self._check_variables(X, self.variables)
        self._verify_specials_variables(self.specials, X.columns)

        self.features_bucket_mapping_ = {}
        self.bucket_tables_ = {}
        self.binners = {}

        for feature in self.variables:

            n_special_bins = 0
            if feature in self.specials.keys():
                special = self.specials[feature]

                n_special_bins = len(special)

                if (self.max_n_bins - n_special_bins) <= 1:
                    raise ValueError(
                        f"max_n_bins must be at least = the number of special bins + 2: set a value "
                        f"max_n_bins>= {n_special_bins+2} (currently max_n_bins={self.max_n_bins})"
                    )

                X_flt, y_flt = self._filter_specials_for_fit(X=X[feature], y=y, specials=special)
            else:
                X_flt = X[feature]
                y_flt = y
                special = {}

            X_flt, y_flt = self._filter_na_for_fit(X=X_flt, y=y_flt)
            # If the specials are excluded, make sure that the bin size is rescaled.
            frac_left = X_flt.shape[0] / X.shape[0]

            # in case everything is a special case, don't fit the tree.
            if frac_left > 0:

                min_bin_size = self.min_bin_size / frac_left

                if min_bin_size > 0.5:
                    min_bin_size = 0.5

                binner = DecisionTreeClassifier(
                    max_leaf_nodes=(self.max_n_bins - n_special_bins),
                    min_samples_leaf=min_bin_size,
                    random_state=self.random_state,
                    **self.kwargs,
                )
                self.binners[feature] = binner
                binner.fit(X_flt.values.reshape(-1, 1), y_flt)

                # Extract fitted boundaries
                splits = np.unique(binner.tree_.threshold[binner.tree_.feature != _tree.TREE_UNDEFINED])

            else:
                splits = []

            self.features_bucket_mapping_[feature] = BucketMapping(
                feature_name=feature,
                type="numerical",
                map=splits,
                right=False,
                specials=special,
                missing_treatment=self.missing_treatment,
            )

            # Calculate the bucket table
            self.bucket_tables_[feature] = build_bucket_table(
                X, y, column=feature, bucket_mapping=self.features_bucket_mapping_.get(feature)
            )

        return self


class OrdinalCategoricalBucketer(BaseBucketer):
    """
    The OrdinalCategoricalEncoder() replaces categories by ordinal numbers.

    Example (0, 1, 2, 3, etc). The numbers are assigned ordered based on the mean of the target
    per category, or assigned in order of frequency, when sort_by_target is False.

    Ordered ordinal encoding: for the variable colour, if the mean of the target
    for blue, red and grey is 0.5, 0.8 and 0.1 respectively, blue is replaced by 2,
    red by 3 and grey by 1. If new data contains unknown labels (f.e. yellow),
    they will be replaced by 0.

    This bucketer:

    - is unsupervised when `encoding_method=='frequency'`: it does not consider
        the target value when fitting the buckets.
    - is supervised when `encoding_method=='ordered'`: it uses
        the target value when fitting the buckets.
    - ignores missing values and passes them through.
    - sets unknown new categories to the category 'other'

    ```python
    from skorecard import datasets
    from skorecard.bucketers import OrdinalCategoricalBucketer

    X, y = datasets.load_uci_credit_card(return_X_y=True)
    bucketer = OrdinalCategoricalBucketer(variables=['EDUCATION'])
    bucketer.fit_transform(X)
    bucketer = OrdinalCategoricalBucketer(max_n_categories=2, variables=['EDUCATION'])
    bucketer.fit_transform(X, y)
    ```

    Credits: Code & ideas adapted from:

    - feature_engine.categorical_encoders.OrdinalCategoricalEncoder
    - feature_engine.categorical_encoders.RareLabelCategoricalEncoder

    """

    def __init__(
        self,
        tol=0.05,
        max_n_categories=None,
        variables=[],
        specials={},
        encoding_method="frequency",
        missing_treatment="separate",
    ):
        """Init the class.

        Args:
            tol (float): the minimum frequency a label should have to be considered frequent.
                Categories with frequencies lower than tol will be grouped together (in the highest ).
            max_n_categories (int): the maximum number of categories that should be considered frequent.
                If None, all categories with frequency above the tolerance (tol) will be
                considered.
            variables (list): The features to bucket. Uses all features if not defined.
            specials: (nested) dictionary of special values that require their own binning.
                The dictionary has the following format:
                 {"<column name>" : {"name of special bucket" : <list with 1 or more values>}}
                For every feature that needs a special value, a dictionary must be passed as value.
                This dictionary contains a name of a bucket (key) and an array of unique values that should be put
                in that bucket.
                When special values are defined, they are not considered in the fitting procedure.
            encoding_method (string): encoding method.
                - "frequency" (default): orders the buckets based on the frequency of observations in the bucket.
                    The lower the number of the bucket the most frequent are the observations in that bucket.
                - "ordered": orders the buckets based on the average class 1 rate in the bucket.
                    The lower the number of the bucket the lower the fraction of class 1 in that bucket.
            missing_treatment: Defines how we treat the missing values present in the data.
                If a string, it must be in ['separate', 'risky', 'frequent']
                separate: Missing values get put in their separate bucket
                risky: todo
                frequent: todo

                If a dict, it must be of the following format:
                {"<column name>": <bucket_number>}
                This bucket number is where we will put the missing values.
        """
        assert isinstance(variables, list)
        self._is_allowed_missing_treatment(missing_treatment)

        if tol < 0 or tol > 1:
            raise ValueError("tol takes values between 0 and 1")

        if max_n_categories is not None:
            if max_n_categories < 0 or not isinstance(max_n_categories, int):
                raise ValueError("max_n_categories takes only positive integer numbers")

        self.tol = tol
        self.max_n_categories = max_n_categories
        self.variables = variables
        self.specials = specials
        self.encoding_method = encoding_method
        self.missing_treatment = missing_treatment

    def fit(self, X, y=None):
        """Init the class."""
        X = self._is_dataframe(X)
        self.variables = self._check_variables(X, self.variables)
        self._verify_specials_variables(self.specials, X.columns)

        self.features_bucket_mapping_ = {}
        self.bucket_tables_ = {}

        for var in self.variables:

            normalized_counts = None
            # Determine the order of unique values

            if var in self.specials.keys():
                special = self.specials[var]
                X_flt, y_flt = self._filter_specials_for_fit(X=X[var], y=y, specials=special)
            else:
                X_flt, y_flt = X[var], y
                special = {}
            if not (isinstance(y_flt, pd.Series) or isinstance(y_flt, pd.DataFrame)):
                y_flt = pd.Series(y_flt)

            X_flt, y_flt = self._filter_na_for_fit(X=X_flt, y=y_flt)

            X_y = pd.concat([X_flt, y_flt], axis=1)
            X_y.columns = [var, "target"]

            if self.encoding_method == "ordered":
                if y is None:
                    raise ValueError("To use encoding_method=='ordered', y cannot be None.")
                # X_flt["target"] = y_flt
                normalized_counts = X_y[var].value_counts(normalize=True)
                cats = X_y.groupby([var])["target"].mean().sort_values(ascending=True).index
                normalized_counts = normalized_counts[cats]

            elif self.encoding_method == "frequency":
                normalized_counts = X_y[var].value_counts(normalize=True)
            else:

                raise NotImplementedError(
                    f"encoding_method='{self.encoding_method}' not supported. "
                    f"Currently implemented options"
                    f" are 'ordered' or 'frequency' (see doc strings)"
                )

            # Limit number of categories if set.
            normalized_counts = normalized_counts[: self.max_n_categories]

            # Remove less frequent categories
            normalized_counts = normalized_counts[normalized_counts >= self.tol]

            # Determine Ordinal Encoder based on ordered labels
            # Note we start at 1, to be able to encode missings as 0.
            mapping = dict(zip(normalized_counts.index, range(0, len(normalized_counts))))

            self.features_bucket_mapping_[var] = BucketMapping(
                feature_name=var,
                type="categorical",
                map=mapping,
                specials=special,
                missing_treatment=self.missing_treatment,
            )

            # Calculate the bucket table
            self.bucket_tables_[var] = build_bucket_table(
                X, y, column=var, bucket_mapping=self.features_bucket_mapping_.get(var)
            )

        return self


class UserInputBucketer(BaseBucketer):
    """Bucket transformer implementing user-defined boundaries.

    This bucketer:
    - is not fitted, as it depends on user defined input
    - ignores missing values and passes them through.

    ```python
    from skorecard import datasets
    from skorecard.bucketers import AgglomerativeClusteringBucketer, UserInputBucketer

    X, y = datasets.load_uci_credit_card(return_X_y=True)

    ac_bucketer = AgglomerativeClusteringBucketer(n_bins=3, variables=['LIMIT_BAL'])
    ac_bucketer.fit(X)
    mapping = ac_bucketer.features_bucket_mapping_

    ui_bucketer = UserInputBucketer(mapping)
    new_X = ui_bucketer.fit_transform(X)
    assert len(new_X['LIMIT_BAL'].unique()) == 3

    #Map some values to the special buckets
    specials = {
        "LIMIT_BAL":{
            "=50000":[50000],
            "in [20001,30000]":[20000,30000],
            }
    }

    ac_bucketer = AgglomerativeClusteringBucketer(n_bins=3, variables=['LIMIT_BAL'], specials = specials)
    ac_bucketer.fit(X)
    mapping = ac_bucketer.features_bucket_mapping_

    ui_bucketer = UserInputBucketer(mapping)
    new_X = ui_bucketer.fit_transform(X)
    assert len(new_X['LIMIT_BAL'].unique()) == 5
    ```

    """

    def __init__(self, features_bucket_mapping: Union[Dict, FeaturesBucketMapping], variables: List = []) -> None:
        """Initialise the user-defined boundaries with a dictionary.

        Notes:
        - features_bucket_mapping is stored without the trailing underscore (_) because it is not fitted.

        Args:
            features_bucket_mapping (dict): Contains the feature name and boundaries defined for this feature.
                Either dict or FeaturesBucketMapping
            variables (list): The features to bucket. Uses all features in features_bucket_mapping if not defined.
        """
        # Check user defined input for bucketing. If a dict is specified, will auto convert
        if not isinstance(features_bucket_mapping, FeaturesBucketMapping):
            if not isinstance(features_bucket_mapping, dict):
                raise TypeError("'features_bucket_mapping' must be a dict or FeaturesBucketMapping instance")
            self.features_bucket_mapping_ = FeaturesBucketMapping(features_bucket_mapping)
        else:
            self.features_bucket_mapping_ = features_bucket_mapping

        # If user did not specify any variables,
        # use all the variables defined in the features_bucket_mapping
        self.variables = variables
        if variables == []:
            self.variables = list(self.features_bucket_mapping_.maps.keys())

    def fit(self, X, y=None):
        """Init the class."""
        # bucket tables can only be computed on fit().
        # so a user will have to .fit() if she/he wants .plot_buckets() and .bucket_table()
        self.bucket_tables_ = {}
        for feature in self.variables:
            # Calculate the bucket table
            self.bucket_tables_[feature] = build_bucket_table(
                X, y, column=feature, bucket_mapping=self.features_bucket_mapping_.get(feature)
            )
        return self
