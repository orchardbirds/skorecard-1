import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.utils.validation import check_is_fitted

from skorecard.bucket_mapping import FeaturesBucketMapping


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
    from skorecard.pipeline import make_coarse_classing_pipeline
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


def find_coarse_classing_step(pipeline: Pipeline):
    """
    Finds the bucketing (course classing) step in a pipeline object.

    This pipeline needs to have been defined using skorecard.pipeline.make_coarse_classing_pipeline

    Args:
        pipeline (sklearn.pipeline.Pipeline): sklearn pipeline

    Returns:
        index (int): position of bucketing step in pipeline.steps
    """
    # Find the bucketing pipeline step
    bucket_pipes = [s for s in pipeline.steps if getattr(s[1], "name", "") == "bucketing_pipeline"]

    # Raise error if missing
    if len(bucket_pipes) == 0:
        msg = """
        Did not find a bucketing pipeline step. Identity the bucketing pipeline step
        using skorecard.pipeline.make_coarse_classing_pipeline. Example:
        
        ```python
        from skorecard.pipeline import make_coarse_classing_pipeline
        bucket_pipeline = make_coarse_classing_pipeline(
            OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, max_n_bins=10, min_bin_size=0.05),
        )
        ```
        """
        raise AssertionError(msg)

    if len(bucket_pipes) > 1:
        msg = """
        You need to identity only the bucketing step, using skorecard.pipeline.make_coarse_classing_pipeline only once.
        
        Example:
        
        ```python
        from skorecard.pipeline import make_coarse_classing_pipeline
        bucket_pipeline = make_coarse_classing_pipeline(
            OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, max_n_bins=10, min_bin_size=0.05),
        )
        ```
        """
        raise AssertionError(msg)

    index_bucket_pipeline = pipeline.steps.index(bucket_pipes[0])
    return index_bucket_pipeline


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
