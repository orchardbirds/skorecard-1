"""Ideas for improving the app.

- Buttons for running 1d bucket transformers
- Sidebar so you can view report over all features https://dash-bootstrap-components.opensource.faculty.ai/examples/
- datatable https://dash.plotly.com/datatable
- plotly dark theme + dash dark theme? https://plotly.com/python/templates/

```python
from skorecard import datasets
from skorecard.apps import ManualBucketerApp

X, y = datasets.load_uci_credit_card(return_X_y=True)

#app = ManualBucketerApp(X)
# app.run_server(mode="external")
# app.stop_server()
```
"""

import copy

from skorecard.bucket_mapping import BucketMapping
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline

from skorecard.utils.exceptions import NotInstalledError
from skorecard.reporting import plot_bins, bucket_table
from skorecard.pipeline import split_pipeline

# Dash + dependencies
try:
    import dash_core_components as dcc
    import dash_html_components as html
    from dash.dependencies import Input, Output, State
    import dash_table
except ModuleNotFoundError:
    dcc = NotInstalledError("dash_core_components", "dashboard")
    html = NotInstalledError("dash_html_components", "dashboard")
    Input = NotInstalledError("dash", "dashboard")
    Output = NotInstalledError("dash", "dashboard")
    State = NotInstalledError("dash", "dashboard")
    dash_table = NotInstalledError("dash_table", "dashboard")

# JupyterDash
try:
    from jupyter_dash import JupyterDash
except ModuleNotFoundError:
    JupyterDash = NotInstalledError("jupyter-dash", "dashboard")

# Dash Bootstrap
try:
    import dash_bootstrap_components as dbc
except ModuleNotFoundError:
    dbc = NotInstalledError("dash_bootstrap_components", "dashboard")

try:
    import plotly.express as px
except ModuleNotFoundError:
    px = NotInstalledError("plotly", "reporting")


# try:
#     import plotly.figure_factory as ff
# except ModuleNotFoundError:
#     ff = NotInstalledError("plotly", "reporting")


# TODO make this internal to the package
# external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
external_stylesheets = [
    # "https://codepen.io/your-codepen-name/pen/your-pen-identifier.css",
    dbc.themes.BOOTSTRAP
]


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

        app = JupyterDash(__name__, external_stylesheets=external_stylesheets)
        self.app = app

        @app.callback(
            Output("original_boundaries", "children"),
            [Input("input_column", "value")],
        )
        def update_original_boundaries(col):
            return str(self.original_feature_mapping.get(col).map)

        @app.callback(
            Output("updated_boundaries", "children"), Input("bucket_table", "data"), State("input_column", "value")
        )
        def update_updated_boundaries(bucket_table, col):
            return str(self.ui_bucketer.features_bucket_mapping.get(col).map)

        @app.callback(
            Output("prebucket_table", "data"),
            [Input("input_column", "value")],
        )
        def get_prebucket_table(col):
            table = bucket_table(x_original=self.X[col], x_bucketed=self.X_prebucketed[col], y=self.y)
            table = table.rename(columns={"bucket": "pre-bucket"})

            # Apply bucket mapping
            bucket_mapping = self.ui_bucketer.features_bucket_mapping.get(col)
            table["bucket"] = bucket_mapping.transform(table["pre-bucket"])
            return table.to_dict("records")

        @app.callback(
            Output("bucket_table", "data"),
            [
                Input("input_column", "value"),
                Input("prebucket_table", "data"),
            ],
        )
        def get_bucket_table(col, prebucket_table):

            new_buckets = pd.DataFrame()
            new_buckets["pre_buckets"] = [row.get("pre-bucket") for row in prebucket_table]
            new_buckets["buckets"] = [int(row.get("bucket")) for row in prebucket_table]

            bucket_mapping = self.ui_bucketer.features_bucket_mapping.get(col)

            boundaries = determine_boundaries(new_buckets, bucket_mapping)
            self.ui_bucketer.features_bucket_mapping.get(col).map = boundaries

            X_bucketed = make_pipeline(self.prebucketing_pipeline, self.ui_bucketer).transform(self.X)
            table = bucket_table(x_original=self.X_prebucketed[col], x_bucketed=X_bucketed[col], y=self.y)
            return table.to_dict("records")

        # Add the layout
        app.layout = html.Div(
            children=[
                dbc.Row(
                    dbc.Col(
                        html.Div(
                            [
                                html.H2(children="skorecard.ManualBucketerApp"),
                                dcc.Dropdown(
                                    id="input_column",
                                    options=[{"label": o, "value": o} for o in self.X_prebucketed.columns],
                                    value=self.X_prebucketed.columns[0],
                                ),
                            ],
                            style={"width": "20%"},
                        )
                    )
                ),
                dcc.Markdown(id="output-container-range-slider"),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(id="graph-prebucket")),
                        dbc.Col(dcc.Graph(id="graph-bucket")),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    html.H4(children="pre-bucketing table"),
                                    html.P(["Original boundaries: ", html.Code(["1,2,4"], id="original_boundaries")]),
                                    html.P(["Updated boundaries: ", html.Code(["1,2,4"], id="updated_boundaries")]),
                                    dash_table.DataTable(
                                        id="prebucket_table",
                                        style_data={
                                            "whiteSpace": "normal",
                                            "height": "auto",
                                        },
                                        style_cell={
                                            "overflow": "hidden",
                                            "textOverflow": "ellipsis",
                                            "maxWidth": 0,
                                            "textAlign": "center",
                                        },
                                        style_as_list_view=True,
                                        page_size=20,
                                        columns=[
                                            {"name": "pre-bucket", "id": "pre-bucket", "editable": False},
                                            {"name": "min", "id": "min", "editable": False},
                                            {"name": "max", "id": "max", "editable": False},
                                            {"name": "count", "id": "count", "editable": False},
                                            {"name": "bucket", "id": "bucket", "editable": True},
                                        ],
                                        style_data_conditional=[
                                            {
                                                "if": {"column_editable": True},
                                                "backgroundColor": "rgb(46,139,87)",
                                                "color": "white",
                                            }
                                        ],
                                        editable=True,
                                    ),
                                ],
                                style={"padding-left": "1em", "width": "600px"},
                            ),
                            style={"margin-left": "1em"},
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    html.H4(children="bucketing table"),
                                    dash_table.DataTable(
                                        id="bucket_table",
                                        style_data={
                                            "whiteSpace": "normal",
                                            "height": "auto",
                                        },
                                        style_cell={
                                            "overflow": "hidden",
                                            "textOverflow": "ellipsis",
                                            "maxWidth": 0,
                                            "textAlign": "center",
                                        },
                                        style_as_list_view=True,
                                        page_size=20,
                                        columns=[
                                            {"name": "bucket", "id": "bucket"},
                                            {"name": "min", "id": "min"},
                                            {"name": "max", "id": "max"},
                                            {"name": "count", "id": "count"},
                                        ],
                                        editable=False,
                                    ),
                                ],
                                style={"margin-right": "1em", "width": "400px"},
                            ),
                        ),
                    ],
                    no_gutters=False,
                    justify="center",
                ),
            ]
        )

        @app.callback(
            Output("graph-prebucket", "figure"),
            [Input("input_column", "value")],
        )
        def plot_dist(col):
            fig = plot_bins(self.X_prebucketed, col)
            fig.update_layout(transition_duration=50)
            fig.update_layout(showlegend=False)
            fig.update_layout(xaxis_title=col)
            fig.update_layout(title="Pre-bucketed")
            return fig

        @app.callback(
            Output("graph-bucket", "figure"),
            [Input("bucket_table", "data")],
        )
        def plot_dist2(data):

            plotdf = pd.DataFrame(
                {"bucket": [int(row.get("bucket")) for row in data], "counts": [int(row.get("count")) for row in data]}
            )

            fig = px.bar(plotdf, x="bucket", y="counts")
            fig.update_layout(transition_duration=50)
            fig.update_layout(showlegend=False)
            fig.update_layout(xaxis_title="Bucket")
            fig.update_layout(title="Bucketed")
            return fig

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


def determine_boundaries(df: pd.DataFrame, bucket_mapping: BucketMapping) -> list:
    """
    Example.

    ```python
    import pandas as pd
    from skorecard.bucket_mapping import BucketMapping
    df = pd.DataFrame()
    df['pre_buckets'] = [0,1,2,3,4,5,6,7,8,9,10]
    df['buckets'] = [0,0,1,1,2,2,2,3,3,4,5]

    bucket_mapping = BucketMapping('feature1', 'numerical', map = [2,3,4,5])

    determine_boundaries(df, bucket_mapping)
    ```
    """
    assert "pre_buckets" in df.columns
    assert "buckets" in df.columns

    if bucket_mapping.type != "numerical":
        raise NotImplementedError("todo")

    dfg = df.groupby(["buckets"]).agg(["max"])
    dfg.columns = dfg.columns.get_level_values(1)
    boundaries = dfg["max"]
    if bucket_mapping.right is False:
        # the prebuckets are integers
        # So we can safely add 1 to make sure the
        # map includes the right prebuckets
        boundaries += 1

    # Drop the last value,
    # This makes sure outlier values are in the same bucket
    # instead of a new one
    boundaries = list(boundaries)[:-1]

    assert sorted(boundaries) == boundaries, "buckets must be sorted"
    return boundaries


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
    pipeline.predict_proba(X)

    from skorecard.pipeline import UserInputPipeline

    uipipe = UserInputPipeline(pipeline, X, y)

    application = uipipe.mb_app.app.server
    application.run(debug=True)
