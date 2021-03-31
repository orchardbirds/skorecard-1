from skorecard.utils import NotPreBucketedError, NotBucketObjectError
from skorecard.pipeline import get_features_bucket_mapping
from skorecard.apps.app_utils import determine_boundaries
from skorecard.reporting import create_report, plot_bucket_table, plot_prebucket_table, plot_bins
from skorecard.bucketers import UserInputBucketer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted

import copy
import pandas as pd


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


    # bucketing_process.summary() # all vars, and # buckets
    # bucketing_process.bucket_table("varname")
    # bucketing_process.plot_bucket("varname")
    # bucketing_process.prebucket_table("varname")
    # bucketing_process.plot_prebucket("varname")

    ```

    """
    def __init__(self, specials={}):
        """Init the class.

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
        self.specials = specials  # I have no idea why this is needed. Remove it for insane errors
        self.name = "bucketingprocess"
    
    def _check_all_bucketers(self, steps):
        """Checks all bucketing steps are skorecard bucketers.
        
        Args:
            steps: skorecard bucketers
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

        pre-bucket	label	            Count	Count (%)	Non-event	Event	Event Rate	WoE	  IV	bucket
        0	        (-inf, 25000.0)	    479.0	7.98	    300.0	    179.0	37.37	    0.73  0.05	0
        1	        [25000.0, 45000.0)	370.0	6.17	    233.0	    137.0	37.03	    0.71  0.04	1

        Args:
            column: The column we wish to analyse
        
        Returns:
            A pandas dataframe of the format above
        """
        if column not in self.X.columns:
            raise ValueError(f"column {column} not in columns of X {self.X.columns}")

        prebucket_table = self.prebucket_table(column=column)
        new_buckets = pd.DataFrame()
        new_buckets["pre_buckets"] = [prebucket for prebucket in prebucket_table['pre-bucket'].values]
        new_buckets["buckets"] = [int(bucket) for bucket in prebucket_table['bucket'].values]

        bucket_mapping = self.ui_bucketer.features_bucket_mapping.get(column)

        boundaries = determine_boundaries(new_buckets, bucket_mapping)
        self.ui_bucketer.features_bucket_mapping.get(column).map = boundaries

        table = create_report(
            self.X_prebucketed_,
            self.y,
            column=column,
            bucket_mapping=self.ui_bucketer.features_bucket_mapping.get(column),
            display_missing=False,
        )
        table = table.rename(columns={"bucket_id": "bucket"})
        table["Event Rate"] = round(table["Event Rate"] * 100, 2)
        return table

    def prebucket_table(self, column):
        """
        Generates the statistics for the buckets of a particular column. An example is seen below:

        bucket	label	      Count	 Count (%)	Non-event	Event	Event Rate	WoE	    IV
        0	    (-inf, 1.0)	  479	 7.98	    300	        179	    37.37	    0.73	0.05
        1	    [1.0, 2.0)	  370	 6.17	    233	        137	    37.03	    0.71	0.04

        Args:
            column: The column we wish to analyse
        
        Returns:
            A pandas dataframe of the format above
        """
        if column not in self.X.columns:
            raise ValueError(f"column {column} not in columns of X {self.X.columns}")

        table = create_report(
            self.X, self.y, column=column, bucket_mapping=self._features_prebucket_mapping.get(column)
        )

        table["Event Rate"] = round(table["Event Rate"] * 100, 2)
        table = table.rename(columns={"bucket_id": "pre-bucket"})

        # Apply bucket mapping
        bucket_mapping = self.ui_bucketer.features_bucket_mapping.get(column)
        table["bucket"] = bucket_mapping.transform(table["pre-bucket"])
        return table

    def plot_prebucket(self, column):
        """
        Generates the prebucket table and produces a corresponding plotly plot.

        Args:
            column: The column we want to visualise
        
        Returns:
            plotly fig
        """
        return plot_prebucket_table(prebucket_table=self.prebucket_table(column), X=self.X_prebucketed_, y=self.y, column=column)

    def plot_bucket(self, column):
        """
        Args:
            column: The column we want to visualise
        
        Returns:
            plotly fig
        """
        return plot_bucket_table(self.bucket_table(column=column))
    
    def register_prebucketing_pipeline(self, *steps, **kwargs):
        """Helps to identify a (series of) sklearn pipeline steps as the pre-bucketing steps.

        Args:
            *steps: skorecard bucketers or other sklearn transformers (passed to sklearn.pipeline.make_pipeline)

            **kwargs:
                memory: See sklearn.pipeline.make_pipeline
                verbose: See sklearn.pipeline.make_pipeline
                name: Add a attribute to Pipeline with a name
                enforce_all_bucketers: Make sure all steps are skorecard bucketers
        """
        self._check_all_bucketers(steps)
        self.prebucketing_pipeline = make_pipeline(*steps, **kwargs)
        self._remap_specials_pipeline(level="prebucketing")

    def register_bucketing_pipeline(self, *steps, **kwargs):
        """Helps to identify a (series of) sklearn pipeline steps as the bucketing steps.

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
        self.X = X
        self.y = y
        self.X_prebucketed_ = self.prebucketing_pipeline.fit_transform(X, y)

        # find the prebucket features bucket mapping. This is necessary
        # to find the mappings of the specials for the bucketing step.
        self._features_prebucket_mapping = get_features_bucket_mapping(self.prebucketing_pipeline)

        # define
        self._retrieve_special_for_bucketing()

        self._remap_specials_pipeline(level="bucketing")

        self.bucketing_pipeline.fit(self.X_prebucketed_, y)
        self._features_bucket_mapping = get_features_bucket_mapping(self.bucketing_pipeline)

        #Add UI bucketer for report
        self.ui_bucketer = UserInputBucketer(self._features_bucket_mapping)
        self.pipeline = make_pipeline(
            self.prebucketing_pipeline, self.ui_bucketer, self.bucketing_pipeline
        )

        return self

    def _remap_specials_pipeline(self, level="prebucketing"):
        """Add the specials in the prebucketing pipeline.

        Specials are designed to be defined in every bucketer.
        This class passes it in the constructor.
        Therefore, this needs to be remapped to the bucketers in the steps of the pipeline.

        Args:
            level (str, optional): define level at which to map Defaults to "prebucketing".

        Raises:
            ValueError: error raised in case the level argument is not supported.
        """
        if level == "prebucketing":
            steps = self.prebucketing_pipeline.steps
            specials = self._prebucketing_specials
        elif level == "bucketing":
            steps = self.bucketing_pipeline.steps
            specials = self._bucketing_specials
        else:
            raise ValueError("level must be prebucketing or bucketing")
        # map the specials to the prebucketer
        for step in steps:
            # Assign the special variables to all the bucketers
            # in all the steps. This is not ideal, but it does not
            # matter, because the bucketers ignore the special variables
            # if not there.
            # TODO: test this!
            step[1].specials = specials

    def _retrieve_special_for_bucketing(self):
        """Finds the indexes of the specials from the prebucketing step.

        Then it creates a new special dictionary, where it maps the specials
        for the bucketing step to the respective index from the prebucketing
        step.
        """
        self._bucketing_specials = dict()

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
        """Regenerate the feature bucket mapping.

        Generate a feature_bucket_mapping that will take the boundaries from the
        prebucket_pipeline and the final output from the bucket_pipeline.
        """
        raise NotImplementedError("Do this :)")

    def transform(self, X):
        """Transform X through the prebucketing and bucketing pipelines."""
        check_is_fitted(self)
        self.X_prebucketed = self.prebucketing_pipeline.transform(X)
        self.X_bucketed = self.bucketing_pipeline.transform(self.X_prebucketed)

        return self.X_bucketed

#TODO:
# bucketing_process.summary() # all vars, and # buckets

