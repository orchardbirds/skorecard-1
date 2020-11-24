import copy
from skorecard.bucketers.bucketers import OrdinalCategoricalBucketer
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline

from skorecard.utils.exceptions import NotInstalledError, BucketingPipelineError
from skorecard.pipeline import find_bucketing_step, get_features_bucket_mapping
from skorecard.bucketers import UserInputBucketer
from skorecard.apps.app_layout import add_layout
from skorecard.apps.app_callbacks import add_callbacks


# JupyterDash
try:
    from jupyter_dash import JupyterDash
except ModuleNotFoundError:
    JupyterDash = NotInstalledError("jupyter-dash", "dashboard")


class BucketTweakerApp(object):
    """Tweak bucketing in a sklearn pipeline manually using a Dash web app.

    Example:

    ```python
    from skorecard import datasets
    from skorecard.bucketers import DecisionTreeBucketer, OptimalBucketer
    from skorecard.pipeline import make_bucketing_pipeline
    from skorecard.apps import BucketTweakerApp
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
        make_bucketing_pipeline(
            OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, max_n_bins=10, min_bin_size=0.05),
        ),
        OneHotEncoder(),
        LogisticRegression()
    )

    pipeline.fit(X, y)
    tweaker = BucketTweakerApp(pipeline, X, y)
    # tweaker.run_server()
    # tweaker.stop_server()
    tweaker.pipeline # or tweaker.get_pipeline()
    ```
    """

    def __init__(self, pipeline: Pipeline, X: pd.DataFrame, y: np.array) -> None:
        """Setup for being able to run the dash app.

        Args:
            pipeline (Pipeline): fitted sklearn pipeline object
            X (pd.DataFrame): input dataframe
            y (np.array): target array
        """
        assert isinstance(X, pd.DataFrame), "X must be pd.DataFrame"

        # Make sure we don't change instance of input pipeline
        pipeline = copy.deepcopy(pipeline)
        self.X = X
        self.y = y

        # Identify the bucketing steps in the pipeline
        index_prebucket_pipeline = find_bucketing_step(pipeline, identifier="prebucketing_pipeline")
        index_bucket_pipeline = find_bucketing_step(pipeline)

        if index_bucket_pipeline != index_prebucket_pipeline + 1:
            msg = "The prebucketing step (make_prebucket_pipeline) should be defined"
            msg += "right before the bucketing step (make_bucket_pipeline)"
            raise ValueError(msg)

        # Extract the features bucket mapping information
        self.original_prebucket_feature_mapping = copy.deepcopy(
            get_features_bucket_mapping(pipeline[index_prebucket_pipeline])
        )
        self.original_bucket_feature_mapping = copy.deepcopy(
            get_features_bucket_mapping(pipeline[index_bucket_pipeline])
        )

        # Split pipeline into different parts
        self.prebucketing_pipeline = Pipeline(pipeline.steps[:index_bucket_pipeline])
        self.postbucketing_pipeline = Pipeline(pipeline.steps[index_bucket_pipeline + 1 :])

        # Here is the real trick
        # We replace the bucketing pipeline step with a UserInputBucketer
        # Now we can tweak the FeatureMapping in the UserInputBucketer
        # Obviously that means you cannot re-fit,
        # but you shouldn't want/need to if you made manual changes.
        self.ui_bucketer = UserInputBucketer(copy.deepcopy(self.original_bucket_feature_mapping))
        self.pipeline = make_pipeline(self.prebucketing_pipeline, self.ui_bucketer, self.postbucketing_pipeline)

        # Now get the prebucketed features
        self.X_prebucketed = self.prebucketing_pipeline.transform(self.X)

        # Checks on prebucketed data
        assert isinstance(self.X_prebucketed, pd.DataFrame)
        # Prebucketed features should have at most 100 unique values.
        # otherwise app prebinning table is too big.
        for feature in self.X_prebucketed.columns:
            if len(self.X_prebucketed[feature].unique()) > 100:
                raise AssertionError(f"{feature} has >100 values. Did you apply pre-bucketing?")

        # All columns should have a prebucketing step defined
        features_prebucket_mapping = get_features_bucket_mapping(self.prebucketing_pipeline)
        missing_columns = [c for c in X.columns if c not in features_prebucket_mapping.columns]
        if len(missing_columns) > 0:
            raise BucketingPipelineError(
                "The following columns do not have a pre-bucketer assigned: %s" % missing_columns
            )

        # Initialize the Dash app, with layout and callbacks
        self.app = JupyterDash(__name__)
        add_layout(self)
        add_callbacks(self)

    def run_server(self, *args, **kwargs):
        """Start a dash server.

        Passes arguments to app.run_server().

        Note we are using a [jupyterdash](https://medium.com/plotly/introducing-jupyterdash-811f1f57c02e) app,
        which supports 3 different modes:

        - 'external' (default): Start dash server and print URL
        - 'inline': Start dash app inside an Iframe in the jupyter notebook
        - 'jupyterlab': Start dash app as a new tab inside jupyterlab

        Use like `run_server(mode='inline')`
        """
        return self.app.run_server(*args, **kwargs)

    def stop_server(self):
        """Stop a running app server.

        This is handy when you want to stop a server running in a notebook.

        [More info](https://community.plotly.com/t/how-to-shutdown-a-jupyterdash-app-in-external-mode/41292/3)
        """
        self.app._terminate_server_for_port("localhost", 8050)

    def get_pipeline(self):
        """Returns pipeline object."""
        return self.pipeline


# This section is here to help debug the Dash app
# This custom code start the underlying flask server from dash directly
# allowing better debugging in IDE's f.e. using breakpoint()
# Example:
# python -m ipdb -c continue manual_bucketer_app.py
if __name__ == "__main__":

    from skorecard import datasets
    from skorecard.bucketers import DecisionTreeBucketer, OptimalBucketer
    from skorecard.pipeline import make_bucketing_pipeline, make_prebucketing_pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LogisticRegression

    df = datasets.load_uci_credit_card(as_frame=True)
    X = df.drop(columns=["default"])
    y = df["default"]

    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    pipeline = make_pipeline(
        make_prebucketing_pipeline(
            DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05),
            OrdinalCategoricalBucketer(variables=cat_cols),
        ),
        make_bucketing_pipeline(
            OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, max_n_bins=10, min_bin_size=0.05),
        ),
        OneHotEncoder(),
        LogisticRegression(),
    )

    pipeline.fit(X, y)

    tweaker = BucketTweakerApp(pipeline, X, y)
    tweaker.run_server()

    application = tweaker.app.server
    application.run(debug=True)
