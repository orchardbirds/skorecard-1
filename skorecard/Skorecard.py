import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from skorecard.linear_model import LogisticRegression
from skorecard.utils import BucketingPipelineError
from skorecard.pipeline import BucketingProcess
from skorecard.preprocessing import WoeEncoder
from skorecard.bucketers import OptimalBucketer, DecisionTreeBucketer


class Skorecard(
    BaseEstimator, TransformerMixin
):
    """Class to wrap all main package components into one place. 

    Usage example:

    ```python
    from skorecard import Skorecard
    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]
    model = Skorecard(specials={'LIMIT_BAL': {'=400000.0' : [400000.0]}},
                    prebucketing_pipeline=[DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05)],
                    bucketing_pipeline=[
                OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
                OptimalBucketer(variables=cat_cols, variables_type='categorical', max_n_bins=10, min_bin_size=0.05)]
    )

    model.fit(X, y)
    ```

    Alternatively:

    ```python
    from skorecard import Skorecard
    from skorecard.pipeline import BucketingProcess
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
    model = Skorecard(bucketing_process=bucketing_process)
    model.fit(X, y)

    # Details
    model.summary() # all vars, and # buckets
    model.bucket_table("LIMIT_BAL")
    model.plot_bucket("LIMIT_BAL")
    model.prebucket_table("LIMIT_BAL")
    model.plot_prebucket("LIMIT_BAL")
    ```

    """
    def __init__(self,
                 specials={},
                 prebucketing_pipeline=None,
                 bucketing_pipeline=None,
                 bucketing_process=None,
                 estimator=LogisticRegression(),
                 encoder=WoeEncoder()):
        """
        Init the class.

        Args:
            specials: (nested) dictionary of special values that require their own binning.
                      Should not be used if bucketing_process is given.
                      The dictionary has the following format:
                      {"<column name>" : {"name of special bucket" : <list with 1 or more values>}}
                      For every feature that needs a special value, a dictionary must be passed as value.
                      This dictionary contains a name of a bucket (key) and an array of unique values that should be put
                      in that bucket.
                      When special values are defined, they are not considered in the fitting procedure.
            prebucketing_pipeline: list of steps we wish to use in the prebucketing process, such as:
                                   prebucketing_pipeline=[DecisionTreeBucketer(variables=num_cols,
                                                                               max_n_bins=100,
                                                                               min_bin_size=0.05)]
                                   Should not be used if bucketing_process is given.
            bucketing_pipeline: list of steps we wish to use in the bucketing process, such as:
                                [OptimalBucketer(variables=num_cols,
                                                 max_n_bins=10,
                                                 min_bin_size=0.05),
                                 OptimalBucketer(variables=cat_cols,
                                                 variables_type='categorical',
                                                 max_n_bins=10,
                                                 min_bin_size=0.05)]
                                Should not be used if bucketing_process is given.
            bucketing_process: The entire pre-bucketing and bucketing pipelines already configured in a BucketingProcess() object.
                               If this methodology is used, specials and prebucketing_pipeline and bucketing_pipeline must not be used.
            estimator: The sklearn-compatible model we want to use to compute probabilities.
                       By default, this is the skorecard LogisticRegression() object.
            encoder: The sklearn-compatible model we want to encode wth.
                     By default, this is the skorecard WoeEncoder() object.
            
        """
        # Checks that only bucketing_process OR prebucketing + bucketing pipelines are used
        if prebucketing_pipeline is None and bucketing_process is None:
            warnings.warn("prebucketing_pipeline and bucketing_process are undefined, "\
                          "using DecisionTreeBucketer(max_n_bins=100) as prebucketing_pipeline")
            prebucketing_pipeline = [DecisionTreeBucketer(max_n_bins=100)]
        elif prebucketing_pipeline is not None and bucketing_process is not None:
            msg = "Choose between prebucketing_pipeline or bucketing_process."
            raise BucketingPipelineError(msg)
        if bucketing_pipeline is None and bucketing_process is None:
            warnings.warn("bucketing_pipeline and bucketing_process are undefined, " \
                          "using OptimalBucketer(max_n_bins=7) as bucketing_pipeline")
            bucketing_pipeline = [OptimalBucketer(max_n_bins=7)]
        elif prebucketing_pipeline is not None and bucketing_process is not None:
            msg = "Choose between bucketing_pipeline or bucketing_process."
            raise BucketingPipelineError(msg)
        if bucketing_process is not None:
            self.bucketing_process = bucketing_process
        else:
            if type(prebucketing_pipeline) != list:
                msg = "prebucketing_pipeline must be a list"
                raise BucketingPipelineError(msg)
            if type(bucketing_pipeline) != list:
                msg = "bucketing_pipeline must be a list"
                raise BucketingPipelineError(msg)
            self.prebucketing_pipeline = prebucketing_pipeline
            self.bucketing_pipeline = bucketing_pipeline

            self.bucketing_process = BucketingProcess(specials=specials)
            self.bucketing_process.register_prebucketing_pipeline(*self.prebucketing_pipeline)
            self.bucketing_process.register_bucketing_pipeline(*self.bucketing_pipeline)

        self.estimator = estimator
        self.encoder = encoder
        self.pipeline = Pipeline([
        ('bucketing_process', self.bucketing_process),
        ('encoder', self.encoder),
        ('clf', self.estimator)
        ])

    def fit(self, X, y=None):
        """Fit the skorecard pipeline with X, y.

        Args:
            X (pd.DataFrame): [description]
            y ([type], optional): [description]. Defaults to None.
        """
        self.pipeline.fit(X, y)
        return self
    
    def transform(self, X):
        """Transform X through the prebucketing and bucketing pipelines, 
        add estimator predictions in column 'prediction'
        """
        X_trans = self.pipeline.named_steps['bucketing_process'].transform(X)
        X_trans['prediction'] = self.estimator.predict_proba(X_trans)[:, 1]
        return X_trans

    def summary(self):
        return self.bucketing_process.summary()

    def prebucket_table(self, column):
        return self.bucketing_process.prebucket_table(column)
    
    def bucket_table(self, column):
        return self.bucketing_process.bucket_table(column)

    def plot_prebucket(self, column):
        return self.bucketing_process.plot_prebucket(column)

    def plot_bucket(self, column):
        return self.bucketing_process.plot_bucket(column)
