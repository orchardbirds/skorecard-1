import warnings

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted

from skorecard.utils import NotPreBucketedError, NotBucketObjectError
from skorecard.pipeline import get_features_bucket_mapping
from skorecard.bucketers import UserInputBucketer
from skorecard.reporting import build_bucket_table
from skorecard.reporting.report import BucketTableMethod, PreBucketTableMethod
from skorecard.reporting.plotting import PlotBucketMethod, PlotPreBucketMethod

from typing import Dict


class BucketingProcess(
    BaseEstimator, TransformerMixin, BucketTableMethod, PreBucketTableMethod, PlotBucketMethod, PlotPreBucketMethod
):
    """Class to concatenate a prebucketing and bucketing step.

    Usage example:

    ```python
    from skorecard import datasets
    from skorecard.bucketers import DecisionTreeBucketer, OptimalBucketer
    from skorecard.pipeline import BucketingProcess

    df = datasets.load_uci_credit_card(as_frame=True)
    y = df["default"]
    X = df.drop(columns=["default"])

    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    bucketing_process = BucketingProcess(specials={'LIMIT_BAL': {'=400000.0' : [400000.0]}})
    bucketing_process.register_prebucketing_pipeline(
                                DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05),
    )
    bucketing_process.register_bucketing_pipeline(
            OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, variables_type='categorical', max_n_bins=10, min_bin_size=0.05),
    )

    bucketing_process.fit(X, y)

    # Details
    bucketing_process.summary() # all vars, and # buckets
    bucketing_process.bucket_table("LIMIT_BAL")
    bucketing_process.plot_bucket("LIMIT_BAL")
    bucketing_process.prebucket_table("LIMIT_BAL")
    bucketing_process.plot_prebucket("LIMIT_BAL")
    ```

    """

    def __init__(self, specials={}):
        """
        Init the class.

        Args:
            specials: (nested) dictionary of special values that require their own binning.
                The dictionary has the following format:
                 {"<column name>" : {"name of special bucket" : <list with 1 or more values>}}
                For every feature that needs a special value, a dictionary must be passed as value.
                This dictionary contains a name of a bucket (key) and an array of unique values that should be put
                in that bucket.
                When special values are defined, they are not considered in the fitting procedure.
        """
        self.prebucketing_pipeline = None
        self._prebucketing_specials = specials
        self._bucketing_specials = dict()  # will be determined later.
        self.name = "bucketingprocess"  # to be able to identity the bucketingprocess in a pipeline

        # self.specials = specials  # I have no idea why this is needed. Remove it for insane errors

    def _check_all_bucketers(self, steps):
        """
        Ensure all specified bucketing steps are skorecard bucketers.

        Args:
            steps: steps in a scikitlearn pipeline
        """
        for step in steps:
            msg = "All bucketing steps must be skorecard bucketers"
            if "skorecard.bucketers" not in str(type(step)):
                raise NotBucketObjectError(msg)

    def register_prebucketing_pipeline(self, *steps, **kwargs):
        """
        Helps to identify a (series of) sklearn pipeline steps as the pre-bucketing steps.

        Args:
            *steps: skorecard bucketers or other sklearn transformers (passed to sklearn.pipeline.make_pipeline)

            **kwargs:
                memory: See sklearn.pipeline.make_pipeline
                verbose: See sklearn.pipeline.make_pipeline
                name: Add a attribute to Pipeline with a name
                enforce_all_bucketers: Make sure all steps are skorecard bucketers
        """
        self._check_all_bucketers(steps)
        # Add specials to all bucketers
        for step in steps:
            if type(step) != tuple:
                step.specials = self._prebucketing_specials
            else:
                step[1].specials = self._prebucketing_specials

        self.prebucketing_pipeline = make_pipeline(*steps, **kwargs)

    def register_bucketing_pipeline(self, *steps, **kwargs):
        """
        Helps to identify a (series of) sklearn pipeline steps as the bucketing steps.

        Args:
            *steps: skorecard bucketers or other sklearn transformers (passed to sklearn.pipeline.make_pipeline)

            **kwargs:
                memory: See sklearn.pipeline.make_pipeline
                verbose: See sklearn.pipeline.make_pipeline
                name: Add a attribute to Pipeline with a name
                enforce_all_bucketers: Make sure all steps are skorecard bucketers
        """
        if not self.prebucketing_pipeline:
            msg = "You need to register a prebucketing pipeline. Please use register_prebucketing_pipeline() first."
            raise NotPreBucketedError(msg)
        self._check_all_bucketers(steps)

        self.bucketing_pipeline = make_pipeline(*steps, **kwargs)

    def fit(self, X, y=None):
        """Fit the prebucketing and bucketing pipeline with X, y.

        Args:
            X (pd.DataFrame): [description]
            y ([type], optional): [description]. Defaults to None.
        """
        # Fit the prebucketing pipeline
        X_prebucketed_ = self.prebucketing_pipeline.fit_transform(X, y)
        assert isinstance(X_prebucketed_, pd.DataFrame)
        self.features_prebucket_mapping_ = get_features_bucket_mapping(self.prebucketing_pipeline)

        # Calculate the prebucket tables.
        self.prebucket_tables_ = dict()
        for column in X.columns:
            if column in self.features_prebucket_mapping_.maps.keys():
                self.prebucket_tables_[column] = build_bucket_table(
                    X, y, column=column, bucket_mapping=self.features_prebucket_mapping_.get(column)
                )

        # Find the new bucket numbers of the specials after prebucketing,
        for var, var_specials in self._prebucketing_specials.items():

            bucket_labels = self.features_prebucket_mapping_.get(var).labels
            new_specials = find_remapped_specials(bucket_labels, var_specials)
            if len(new_specials):
                self._bucketing_specials[var] = new_specials

        # Then assign the new specials to all bucketers in the bucketing pipeline
        for step in self.bucketing_pipeline.steps:
            if type(step) != tuple:
                step.specials = self._bucketing_specials
            else:
                step[1].specials = self._bucketing_specials

        # Fit the prebucketing pipeline
        # And save the bucket mapping
        self.bucketing_pipeline.fit(X_prebucketed_, y)
        self.features_bucket_mapping_ = get_features_bucket_mapping(self.bucketing_pipeline)
        # and calculate the bucket tables.
        self.bucket_tables_ = dict()
        for column in X.columns:
            if column in self.features_bucket_mapping_.maps.keys():
                self.bucket_tables_[column] = build_bucket_table(
                    X_prebucketed_, y, column=column, bucket_mapping=self.features_bucket_mapping_.get(column)
                )

        return self

    def _set_bucket_mapping(self, features_bucket_mapping, X_prebucketed, y):
        """
        Replace the bucket mapping in the bucketing_pipeline.

        This is meant for use internally in the dash app, where we manually edit
        bucketingprocess.features_bucket_mapping_.

        To be able to update the bucketingprocess, use something like:

        >>> X_prebucketed = bucketingprocess.prebucket_pipeline.transform(X)
        >>> feature_bucket_mapping # your edited bucketingprocess.features_bucket_mapping_
        >>> bucketingprocess._set_bucket_mapping(feature_bucket_mapping, X_prebucketed, y)
        """
        # Step 1: replace the bucketing pipeline with a UI bucketer that uses the new mapping
        self.bucketing_pipeline = UserInputBucketer(features_bucket_mapping)
        self.features_bucket_mapping_ = features_bucket_mapping

        # Step 2: Recalculate the bucket tables
        self.bucket_tables_ = dict()
        for column in X_prebucketed.columns:
            if column in self.features_bucket_mapping_.maps.keys():
                self.bucket_tables_[column] = build_bucket_table(
                    X_prebucketed, y, column=column, bucket_mapping=self.features_bucket_mapping_.get(column)
                )

    def transform(self, X):
        """Transform X through the prebucketing and bucketing pipelines."""
        check_is_fitted(self)
        X_prebucketed = self.prebucketing_pipeline.transform(X)
        return self.bucketing_pipeline.transform(X_prebucketed)

    def summary(self):
        """
        Generates a summary table for columns passed to `.fit()`.

        The format is the following:

        column    | num_prebuckets | num_buckets | dtype
        -------------------------------------------------
        LIMIT_BAL |      15        |     10      | float64
        BILL_AMT1 |      15        |     6       | float64
        """  # noqa
        check_is_fitted(self)
        columns = []
        num_prebuckets = []
        num_buckets = []
        dtypes = []
        for col in self.X_prebucketed_.columns:
            columns.append(col)
            # In case the column was never (pre)-bucketed
            try:
                prebucket_number = len(self.prebucket_table(col)["pre-bucket"].unique())
            except KeyError:
                warnings.warn(f"Column {col} not pre-bucketed")
                prebucket_number = "not_bucketed"
            try:
                bucket_number = len(self.bucket_table(col)["bucket"].unique())
            except KeyError:
                warnings.warn(f"Column {col} not bucketed")
                bucket_number = "not_bucketed"
            num_prebuckets.append(prebucket_number)
            num_buckets.append(bucket_number)
            dtypes.append(self.X[col].dtype)

        return pd.DataFrame(
            {"column": columns, "num_prebuckets": num_prebuckets, "num_buckets": num_buckets, "dtype": dtypes}
        )


def find_remapped_specials(bucket_labels: Dict, var_specials: Dict) -> Dict:
    """
    Remaps the specials after the prebucketing process.

    Special values will have changed.

    Args:
        bucket_labels (dict): The label for each unique bucket of a variable
        var_specials (dict): The specials for a variable, if any.
    """
    if bucket_labels is None or var_specials is None:
        return {}

    new_specials = {}
    for label in var_specials.keys():
        for bucket, bucket_label in bucket_labels.items():
            if bucket_label == f"Special: {label}":
                new_specials[label] = [bucket]

    return new_specials
