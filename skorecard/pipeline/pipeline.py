import logging
import copy

from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.utils.validation import check_is_fitted

from skorecard.bucket_mapping import FeaturesBucketMapping
from skorecard.bucketers import UserInputBucketer


class KeepPandas(BaseEstimator, TransformerMixin):
    """Keep pandas dataframe in a sklearn pipeline.

    This is a helper class to turn sklearn transformations back to pandas.

    !!! warning
        You should only use `KeepPandas()` when you know for sure `sklearn`
        did not change the order of your columns.

    ```python
    from skorecard.pipeline import KeepPandas
    from skorecard import datasets
    from skorecard.bucketers import EqualWidthBucketer

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = datasets.load_uci_credit_card(return_X_y=True)

    bucket_pipeline = make_pipeline(
        KeepPandas(StandardScaler()),
        EqualWidthBucketer(bins=5, variables=['LIMIT_BAL', 'BILL_AMT1']),
    )
    bucket_pipeline.fit_transform(X, y)
    ```
    """

    def __init__(self, transformer):
        """Initialize."""
        self.transformer = transformer

        # Warn if there is a chance order of columns are changed
        if isinstance(transformer, Pipeline):
            for step in _get_all_steps(transformer):
                self._check_for_column_transformer(step)
        else:
            self._check_for_column_transformer(transformer)

    def __repr__(self):
        """String representation."""
        return self.transformer.__repr__()

    def _check_for_column_transformer(self, obj):
        msg = "sklearn.compose.ColumnTransformer can change the order of columns"
        msg += ", be very careful when using with KeepPandas()"
        if type(obj).__name__ == "ColumnTransformer":
            logging.warning(msg)

    def fit(self, X, y=None, *args, **kwargs):
        """Fit estimator."""
        assert isinstance(X, pd.DataFrame)
        self.columns_ = list(X.columns)
        self.transformer.fit(X, y, *args, **kwargs)
        return self

    def transform(self, X, *args, **kwargs):
        """Transform X."""
        check_is_fitted(self)
        new_X = self.transformer.transform(X, *args, **kwargs)
        return pd.DataFrame(new_X, columns=self.columns_)

    def get_feature_names(self):
        """Return estimator feature names."""
        check_is_fitted(self)
        return self.columns_


def make_coarse_classing_pipeline(*steps, **kwargs):
    """Identity sklearn pipeline steps as coarse classing.

    Very simple wrapper of sklearn.pipeline.make_pipeline()

    ```python
    from skorecard import datasets
    from skorecard.bucketers import DecisionTreeBucketer, OptimalBucketer
    from skorecard.pipeline import make_coarse_classing_pipeline, tweak_buckets
    from sklearn.pipeline import make_pipeline

    df = datasets.load_uci_credit_card(as_frame=True)
    y = df["default"]
    X = df.drop(columns=["default"])

    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    pipeline = make_pipeline(
        DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05),
        make_coarse_classing_pipeline(
            OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, max_n_bins=10, min_bin_size=0.05),
        )
    )

    pipeline.fit(X, y)
    ```
    """
    for step in steps:
        msg = "All coarse classing steps must be skorecard bucketers"
        assert "skorecard.bucketers" in str(type(step)), msg

    pipeline = make_pipeline(*steps, **kwargs)

    # Identifier, to make it easy to find this Pipeline step in a pipeline structure
    pipeline.name = "bucketing_pipeline"

    return pipeline


def tweak_buckets(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    """Tweak the bucket manually.

    ```python
    from skorecard import datasets
    from skorecard.bucketers import DecisionTreeBucketer, OptimalBucketer
    from skorecard.pipeline import make_coarse_classing_pipeline, tweak_buckets
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LogisticRegression

    df = datasets.load_uci_credit_card(as_frame=True)
    X = df.drop(columns=["default"])
    y = df["default"]

    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    pipeline = make_pipeline(
        DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05),
        make_coarse_classing_pipeline(
            OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, max_n_bins=10, min_bin_size=0.05),
        ),
        OneHotEncoder(),
        LogisticRegression()
    )

    pipeline.fit(X, y)
    pipeline.predict_proba(X)
    # pipe2 = tweak_buckets(pipeline, X, y) # not run - don't start server
    ```
    """
    # Copy the pipeline
    pipe = copy.deepcopy(pipe)

    # Find the bucketing pipeline step
    bucket_pipes = [s for s in pipe.steps if getattr(s[1], "name", "") == "bucketing_pipeline"]

    # Find the bucketing pipeline step
    if len(bucket_pipes) == 0:
        msg = """
        Did not find a bucketing pipeline step. Identity the bucketing pipeline step
        using skorecard.pipeline.make_coarse_classing_pipeline. Example:
        
        ```python
        from skorecard.pipeline import set_as_bucketing_step
        bucket_pipeline = make_coarse_classing_pipeline(
            OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, max_n_bins=10, min_bin_size=0.05),
        )
        ```
        """
        raise AssertionError(msg)

    if len(bucket_pipes) > 1:
        msg = """
        You need to identity only the bucketing step. You can combine multiple bucketing steps
        using sklearn.pipeline.make_pipeline(). Example:
        
        ```python
        bucket_pipeline = make_coarse_classing_pipeline(
            OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, max_n_bins=10, min_bin_size=0.05),
        )
        ```
        """
        raise AssertionError(msg)

    index_bucket_pipeline = pipe.steps.index(bucket_pipes[0])

    # Get the prebucketed, prepared features.
    try:
        X_prebucketed = Pipeline(pipe.steps[:index_bucket_pipeline]).transform(X)
    except NotFittedError:
        pipe.fit(X, y)
        X_prebucketed = Pipeline(pipe.steps[:index_bucket_pipeline]).transform(X)

    # Checks on prebucketed data
    assert isinstance(X_prebucketed, pd.DataFrame)
    # Prebucketed features should have at most 100 unique values.
    # otherwise app prebinning table is too big.
    for feature in X_prebucketed.columns:
        if len(X_prebucketed[feature].unique()) > 100:
            raise AssertionError(f"{feature} has >100 values. Did you pre-bucket?")

    # Save the reference to feature_bucket_mapping_ instance
    # This way, we can easily change the mapping for a feature manually
    features_bucket_mapping = get_features_bucket_mapping(pipe[index_bucket_pipeline])

    # Overwrite the bucketering pipeline with a UserInputBucketer
    # This is the real 'trick'
    # as it allows us to update with a
    # (potentially tweaked) feature_bucket_mapping
    pipe.steps.pop(index_bucket_pipeline)
    ui_bucketer = UserInputBucketer(features_bucket_mapping)
    pipe.steps.insert(index_bucket_pipeline, ["manual_coarse_classing", ui_bucketer])

    # ui_bucketer.features_bucket_mapping.get('LIMIT_BAL').map = [1,2,3,4,5]

    # def find_ui_bucketer(pipe):
    #     for s in pipe.steps:

    # Start app
    # app.stop_server()
    # This import is here to prevent a circular import
    from skorecard.apps import ManualBucketerApp

    app = ManualBucketerApp(pipe, X, y)
    app.run_server()

    return pipe
    # pipe[index_bucket_pipeline].pipeline.features_bucket_mapping_ = <from our app>
    # bucketed_X = pipe.transform(X)
    # binning_table(bucketed_X, y)


def split_pipeline(pipe: Pipeline) -> Tuple[Pipeline, UserInputBucketer, Pipeline]:
    """Splits a pipeline into three parts.

    1) prebucketing (includes preprocessing)
    2) bucketing by a UserInputBucketer
    3) postbucketing (includesany postprocessing like WoeEncoder and modelling steps)

    Args:
        pipe (Pipeline): [description]

    Example:

    ```python
    1==1
    ```

    Returns:
        Tuple[Pipeline, UserInputBucketer, Pipeline]: [description]
    """
    pipe = copy.deepcopy(pipe)

    prebucketing = []
    ui_bucketer = None
    postbucketing = []

    for step in pipe.steps:
        if type(step[1]) == UserInputBucketer:
            ui_bucketer = step[1]
        else:
            if ui_bucketer:
                postbucketing.append(step)
            else:
                prebucketing.append(step)

    assert (
        prebucketing
    ), "pipeline does not have any preprocessing steps before the UserInputBucketer. Did you do pre-binning?"
    assert ui_bucketer, "pipeline does not have a step with a UserInputBucketer"
    assert postbucketing, "pipeline does not have any steps after the UserInputBucketer"

    return Pipeline(prebucketing), ui_bucketer, Pipeline(postbucketing)


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
    assert isinstance(pipe, BaseEstimator)

    features_bucket_mapping = {}
    for step in _get_all_steps(pipe):
        check_is_fitted(step)
        if hasattr(step, "features_bucket_mapping_"):
            features_bucket_mapping.update(step.features_bucket_mapping_)

    assert (
        len(features_bucket_mapping) > 0
    ), "pipeline does not have any fitted skorecard bucketer. Update the pipeline or fit(X,y) first"
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
