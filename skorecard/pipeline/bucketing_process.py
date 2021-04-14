from skorecard.utils import NotPreBucketedError, NotBucketObjectError
from skorecard.pipeline import get_features_bucket_mapping
from skorecard.reporting import create_report, plot_bucket_table, plot_prebucket_table
from skorecard.bucketers import UserInputBucketer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted

import pandas as pd

import warnings


class BucketingProcess(BaseEstimator, TransformerMixin):
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

        self.specials = specials  # I have no idea why this is needed. Remove it for insane errors

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

    def bucket_table(self, column):
        """
        Generates the statistics for the buckets of a particular column.

        The pre-buckets are matched to the post-buckets, so that the user has a much clearer understanding of how
        the BucketingProcess ends up with the final buckets.
        An example is seen below:

        bucket     | label              | Count | Count (%) | Non-event | Event | Event Rate | WoE |  IV
        ---------------------------------------------------------------------------------------------------
        0          | (-inf, 25000.0)    | 479.0 | 7.98      | 300.0     | 179.0 | 37.37      | 0.73 | 0.05
        1          | [25000.0, 45000.0) | 370.0 | 6.17      | 233.0     | 137.0 | 37.03      | 0.71 | 0.04

        Args:
            column: The column we wish to analyse

        Returns:
            df (pd.DataFrame): A pandas dataframe of the format above
        """  # noqa
        check_is_fitted(self)
        if column not in self.bucket_tables_.keys():
            raise ValueError(f"column '{column}' was not part of the bucketingprocess")

        table = self.bucket_tables_.get(column)
        table = table.rename(columns={"bucket_id": "bucket"})
        table["Event Rate"] = round(table["Event Rate"] * 100, 2)
        return table

    def prebucket_table(self, column):
        """
        Generates the statistics for the buckets of a particular column.

        An example is seen below:

        pre-bucket | label      | Count | Count (%) | Non-event | Event | Event Rate | WoE   | IV  | bucket
        ---------------------------------------------------------------------------------------------------
        0          | (-inf, 1.0)| 479   | 7.98      | 300       | 179   |  37.37     |  0.73 | 0.05 | 0
        1          | [1.0, 2.0) | 370   | 6.17      | 233       | 137   |  37.03     |  0.71 | 0.04 | 0

        Args:
            column: The column we wish to analyse

        Returns:
            df (pd.DataFrame): A pandas dataframe of the format above
        """  # noqa
        check_is_fitted(self)
        if column not in self.prebucket_tables_.keys():
            raise ValueError(f"column '{column}' was not part of the pre-bucketing process")

        table = self.prebucket_tables_.get(column)

        table["Event Rate"] = round(table["Event Rate"] * 100, 2)
        table = table.rename(columns={"bucket_id": "pre-bucket"})

        # Apply bucket mapping
        bucket_mapping = self._features_bucket_mapping.get(column)
        table["bucket"] = bucket_mapping.transform(table["pre-bucket"])
        return table

    def plot_prebucket(self, column, format=None, scale=None, width=None, height=None):
        """
        Generates the prebucket table and produces a corresponding plotly plot.

        Args:
            column: The column we want to visualise
            format: The format of the image, such as 'png'. The default None returns a plotly image.
            scale: If format is specified, the scale of the image
            width: If format is specified, the width of the image
            height: If format is specified, the image of the image

        Returns:
            plot: plotly fig
        """
        check_is_fitted(self)
        return plot_prebucket_table(
            prebucket_table=self.prebucket_table(column),
            X=self.X_prebucketed,
            y=self.y,
            column=column,
            format=format,
            scale=scale,
            width=width,
            height=height,
        )

    def plot_bucket(self, column, format=None, scale=None, width=None, height=None):
        """
        Plot the buckets.

        Args:
            column: The column we want to visualise
            format: The format of the image, such as 'png'. The default None returns a plotly image.
            scale: If format is specified, the scale of the image
            width: If format is specified, the width of the image
            height: If format is specified, the image of the image

        Returns:
            plot: plotly fig
        """
        check_is_fitted(self)
        return plot_bucket_table(
            self.bucket_table(column=column), format=format, scale=scale, width=width, height=height
        )

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
        self._features_prebucket_mapping = get_features_bucket_mapping(self.prebucketing_pipeline)
        # and calculate the prebucket tables.
        self.prebucket_tables_ = dict()
        for column in X.columns:
            if column in self._features_prebucket_mapping.maps.keys():
                self.prebucket_tables_[column] = create_report(
                    X, y, column=column, bucket_mapping=self._features_prebucket_mapping.get(column)
                )

        # TODO: these need to be removed later
        # they are only used in .plot_prebucket_table()
        # and that should be refactored because all info is already saved in self.prebucket_tables
        self.y = y
        self.X_prebucketed = X_prebucketed_

        # Find the new bucket numbers of the specials after prebucketing,
        # and set self._bucketing_specials
        self._retrieve_special_for_bucketing()
        # Then assign the new specials to all bucketers in the bucketing pipeline
        for step in self.bucketing_pipeline.steps:
            if type(step) != tuple:
                step.specials = self._bucketing_specials
            else:
                step[1].specials = self._bucketing_specials

        # Fit the prebucketing pipeline
        # And save the bucket mapping
        self.bucketing_pipeline.fit(X_prebucketed_, y)
        self._features_bucket_mapping = get_features_bucket_mapping(self.bucketing_pipeline)
        # and calculate the bucket tables.
        self.bucket_tables_ = dict()
        for column in X.columns:
            if column in self._features_bucket_mapping.maps.keys():
                self.bucket_tables_[column] = create_report(
                    X_prebucketed_, y, column=column, bucket_mapping=self._features_bucket_mapping.get(column)
                )

        return self

    def _set_bucket_mapping(self, features_bucket_mapping, X_prebucketed, y):
        """
        Replace the bucket mapping in the bucketing_pipeline.

        This is meant for use internally in the dash app, where we manually edit
        bucketingprocess._features_bucket_mapping.

        To be able to update the bucketingprocess, use something like:
        X_prebucketed = bucketingprocess.prebucket_pipeline.transform(X)
        feature_bucket_mapping # your edited bucketingprocess._features_bucket_mapping
        bucketingprocess._set_bucket_mapping(feature_bucket_mapping, X_prebucketed, y)
        """
        # Step 1: replace the bucketing pipeline with a UI bucketer that uses the new mapping
        self.bucketing_pipeline = UserInputBucketer(features_bucket_mapping)
        self._features_bucket_mapping = features_bucket_mapping

        # Step 2: Recalculate the bucket tables
        self.bucket_tables_ = dict()
        for column in X_prebucketed.columns:
            if column in self._features_bucket_mapping.maps.keys():
                self.bucket_tables_[column] = create_report(
                    X_prebucketed, y, column=column, bucket_mapping=self._features_bucket_mapping.get(column)
                )

    def _retrieve_special_for_bucketing(self):
        """
        Finds the indexes of the specials from the prebucketing step.

        Then it creates a new special dictionary, where it maps the specials
        for the bucketing step to the respective index from the prebucketing
        step.
        """
        for var in self._prebucketing_specials.keys():
            feats_preb_map = self._features_prebucket_mapping.maps[var]
            # this finds the maximum index within the
            # the prebucketed maps.  The next index is reserved for the
            # missing value.
            max_val = len(feats_preb_map.map)
            missing_index = max_val + 1

            # Define the bucketing specials by using the same
            # keys as in the prebucketing specials.
            # It then assigns to this keys the index that the
            # prebucketing step maps it to.
            # This assures that the information propagates between
            # the two steps.
            self._bucketing_specials[var] = {
                key: [missing_index + 1 + key_ix]  # the key
                for key_ix, key in enumerate(self._prebucketing_specials[var].keys())
            }

    def _remap_feature_bucket_mapping(self):
        """
        Regenerate the feature bucket mapping.

        Generate a feature_bucket_mapping that will take the boundaries from the
        prebucket_pipeline and the final output from the bucket_pipeline.
        """
        raise NotImplementedError("Do this :)")

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
