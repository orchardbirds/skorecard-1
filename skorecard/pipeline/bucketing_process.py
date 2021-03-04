from .pipeline import make_prebucketing_pipeline, make_bucketing_pipeline
from ..utils import NotPreBucketedError

from sklearn.base import BaseEstimator, TransformerMixin

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

    bucketing_process = BucketingProcess(specials={'column': {'label' : 'value'}})
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
        self.prebucketing_pipeline = None
        self.specials = specials

    def register_prebucketing_pipeline(self, *steps, **kwargs):
        """Helps to identify a (series of)sklearn pipeline steps as the pre-bucketing steps.

        Args:
            *steps: skorecard bucketers or other sklearn transformers (passed to sklearn.pipeline.make_pipeline)

            **kwargs:
                memory: See sklearn.pipeline.make_pipeline
                verbose: See sklearn.pipeline.make_pipeline
                name: Add a attribute to Pipeline with a name
                enforce_all_bucketers: Make sure all steps are skorecard bucketers
        """
        self.prebucketing_pipeline = make_prebucketing_pipeline(*steps, **kwargs)

    def register_bucketing_pipeline(self, *steps, **kwargs):
        """Helps to identify a (series of)sklearn pipeline steps as the bucketing steps.

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

        self.bucketing_pipeline = make_bucketing_pipeline(*steps, **kwargs)

    def fit(self, X, y=None):
        """Fit the prebucketing and bucketing pipeline with X,y.

        Args:
            X (pd.DataFrame): [description]
            y ([type], optional): [description]. Defaults to None.

        """
        self.X_prebucketed_ = self.prebucketing_pipeline.fit_transform(X, y)

        self.bucketing_pipeline.fit(self.X_prebucketed_, y)

        return self

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


# bucketing_process.summary() # all vars, and # buckets
# bucketing_process.bucket_table("varname")
# bucketing_process.bucket_plot("varname")
# bucketing_process.prebucket_table("varname")
# bucketing_process.prebucket_plot("varname")
# bucketing_process.buckets
# bucketing_process.prebuckets
