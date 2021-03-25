from skorecard.utils import NotPreBucketedError
from skorecard.pipeline import get_features_bucket_mapping

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline

# from sklearn.pipeline import make_pipeline as scikit_make_pipeline
from sklearn.utils.validation import check_is_fitted


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
            OptimalBucketer(variables=cat_cols, max_n_bins=10, min_bin_size=0.05),
    )

    bucketing_process.fit(X, y)

    # bucketing_process.summary() # all vars, and # buckets
    # bucketing_process.bucket_table("varname")
    # bucketing_process.bucket_plot("varname")
    # bucketing_process.prebucket_table("varname")
    # bucketing_process.prebucket_plot("varname")
    # bucketing_process.buckets
    # bucketing_process.prebuckets

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
        # @Dan TODO:
        # Make sure all prebiucketing and bucketing steps are skorecard.
        # for step in steps:
        #     msg = "All bucketing steps must be skorecard bucketers"
        #     assert "skorecard.bucketers" in str(type(step)), msg

        self.prebucketing_pipeline = None
        self._prebucketing_specials = specials
        self.name = "bucketingprocess"

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

        self.bucketing_pipeline = make_pipeline(*steps, **kwargs)

    def fit(self, X, y=None):
        """Fit the prebucketing and bucketing pipeline with X, y.

        Args:
            X (pd.DataFrame): [description]
            y ([type], optional): [description]. Defaults to None.

        """
        self.X_prebucketed_ = self.prebucketing_pipeline.fit_transform(X, y)

        # find the prebucket features bucket mapping. This is necessary
        # to find the mappings of the specials for the bucketing step.
        self._features_prebucket_mapping = get_features_bucket_mapping(self.prebucketing_pipeline)

        # define
        self._retrieve_special_for_bucketing()

        self._remap_specials_pipeline(level="bucketing")

        self.bucketing_pipeline.fit(self.X_prebucketed_, y)
        self._features_bucket_mapping = get_features_bucket_mapping(self.bucketing_pipeline)

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
        self.X_bucketed = self.bucketing_pipeline.transform(X)

        return self.X_bucketed

    @property
    def buckets(self):
        """
        Buckets after bucketing process.
        """
        pass


# bucketing_process.summary() # all vars, and # buckets
# bucketing_process.bucket_table("varname")
# bucketing_process.bucket_plot("varname")
# bucketing_process.prebucket_table("varname")
# bucketing_process.prebucket_plot("varname")
# bucketing_process.buckets
# bucketing_process.prebuckets
