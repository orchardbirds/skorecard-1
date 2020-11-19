import copy
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline

from skorecard.utils.exceptions import NotInstalledError
from skorecard.pipeline import split_pipeline
from skorecard.apps.app_layout import get_layout
from skorecard.apps.app_callbacks import add_callbacks

# JupyterDash
try:
    from jupyter_dash import JupyterDash
except ModuleNotFoundError:
    JupyterDash = NotInstalledError("jupyter-dash", "dashboard")


class ManualBucketerApp(object):
    """Dash App for manual bucketing.

    Class that contains a Dash app
    """

    def __init__(self, pipeline: Pipeline, X: pd.DataFrame, y):
        """Create new dash app.

        Args:
            X (pd.DataFrame): input dataframe
            y (np.array): target array
            features_bucket_mapping: Class with bucketing information for features


        Returns:
            dash: Dash app
        """
        assert isinstance(X, pd.DataFrame), "X must be pd.DataFrame"

        self.pipeline = pipeline
        self.X = X
        self.y = y

        self.prebucketing_pipeline, self.ui_bucketer, self.postbucketing_pipeline = split_pipeline(pipeline)
        self.X_prebucketed = self.prebucketing_pipeline.transform(self.X)

        self.original_feature_mapping = copy.deepcopy(self.ui_bucketer.features_bucket_mapping)

        app = JupyterDash(__name__)
        self.app = app

        # Get columns
        column_options = [{"label": o, "value": o} for o in self.X_prebucketed.columns]
        # Add the layout
        app.layout = get_layout(column_options=column_options)
        # Add the callbacks
        app = add_callbacks(app, self)

    def run_server(self, *args, **kwargs):
        """Start a dash server.

        Passes arguments to app.run_server()
        """
        return self.app.run_server(*args, **kwargs)

    def stop_server(self):
        """Stop a running app server.

        This is handy when you want to stop a server running in a notebook.

        [More info](https://community.plotly.com/t/how-to-shutdown-a-jupyterdash-app-in-external-mode/41292/3)
        """
        self.app._terminate_server_for_port("localhost", 8050)


# This section is here to help debug the Dash app
# This custom code start the underlying flask server from dash directly
# allowing better debugging in IDE's f.e. using breakpoint()
# Example:
# python -m ipdb -c continue manual_bucketer_app.py
if __name__ == "__main__":

    from skorecard import datasets
    from skorecard.bucketers import DecisionTreeBucketer, OptimalBucketer
    from skorecard.pipeline import make_coarse_classing_pipeline
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
        LogisticRegression(),
    )

    pipeline.fit(X, y)

    from skorecard.pipeline import UserInputPipeline

    uipipe = UserInputPipeline(pipeline, X, y)

    application = uipipe.mb_app.app.server
    application.run(debug=True)
