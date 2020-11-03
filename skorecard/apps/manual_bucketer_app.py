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
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline

from skorecard.utils.exceptions import NotInstalledError
from skorecard.reporting import plot_bins, bucket_table
from skorecard.pipeline import split_pipeline
from skorecard.apps.app_utils import determine_boundaries, perc_data_bars, colorize_cell, get_bucket_colors

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
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
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
            [Output("bucket_table", "data"), Output("pre-bucket-error", "children")],
            [Input("input_column", "value"), Input("prebucket_table", "data")],
        )
        def get_bucket_table(col, prebucket_table):

            new_buckets = pd.DataFrame()
            new_buckets["pre_buckets"] = [row.get("pre-bucket") for row in prebucket_table]
            new_buckets["buckets"] = [int(row.get("bucket")) for row in prebucket_table]

            # Explicit error handling
            if all(new_buckets["buckets"].sort_values().values == new_buckets["buckets"].values):
                error = []
            else:
                error = dbc.Alert("The buckets most be in ascending order!", color="danger")
                return None, error

            bucket_mapping = self.ui_bucketer.features_bucket_mapping.get(col)

            boundaries = determine_boundaries(new_buckets, bucket_mapping)
            self.ui_bucketer.features_bucket_mapping.get(col).map = boundaries

            X_bucketed = make_pipeline(self.prebucketing_pipeline, self.ui_bucketer).transform(self.X)
            table = bucket_table(
                x_original=self.X_prebucketed[col], x_bucketed=X_bucketed[col], y=self.y, bucket_mapping=bucket_mapping
            )
            return table.to_dict("records"), error

        # Add the layout
        app.layout = html.Div(
            children=[
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    html.H3(
                                        [
                                            "skorecard | Bucketing App",
                                            dbc.Badge("AUC: 0.98", className="ml-1", id="auc-badge"),
                                        ]
                                    ),
                                ]
                            )
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    # html.Div("hello", style={'display' : 'inline-block'}),
                                    # html.Div("hello", style={'display' : 'inline-block'}),
                                    html.Div(
                                        dcc.Dropdown(
                                            id="input_column",
                                            options=[{"label": o, "value": o} for o in self.X_prebucketed.columns],
                                            value=self.X_prebucketed.columns[0],
                                            style={
                                                "max-width": "300px",
                                                "min-width": "250px",
                                                "background-color": "#ededed",
                                            },
                                        ),
                                        style={"display": "inline-block", "float": "right"},
                                    ),
                                    # html.Div(dbc.Button("Regular button", className="mr-1"),
                                    # style={'display' : 'inline-block'}),
                                ]
                            )
                        ),
                    ],
                    style={"padding-bottom": "1em"},
                ),
                dbc.Row([dbc.Col(dcc.Graph(id="graph-prebucket")), dbc.Col(dcc.Graph(id="graph-bucket"))]),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    html.H4(children="pre-bucketing table"),
                                    html.Div(children=[], id="pre-bucket-error"),
                                    dash_table.DataTable(
                                        id="prebucket_table",
                                        style_data={"whiteSpace": "normal", "height": "auto"},
                                        style_cell={
                                            "height": "auto",
                                            "overflow": "hidden",
                                            "textOverflow": "ellipsis",
                                            "maxWidth": 0,
                                            "textAlign": "center",
                                        },
                                        style_cell_conditional=[
                                            {"if": {"column_id": "range"}, "width": "180px"},
                                        ],
                                        style_as_list_view=True,
                                        page_size=20,
                                        columns=[
                                            {"name": "pre-bucket", "id": "pre-bucket", "editable": False},
                                            {"name": "range", "id": "range", "editable": False},
                                            {"name": "count", "id": "count", "editable": False},
                                            {"name": "count %", "id": "count %", "editable": False},
                                            {"name": "Non-event", "id": "Non-event", "editable": False},
                                            {"name": "Event", "id": "Event", "editable": False},
                                            {"name": "Event Rate", "id": "Event Rate", "editable": False},
                                            {"name": "WoE", "id": "WoE", "editable": False},
                                            {"name": "IV", "id": "IV", "editable": False},
                                            {"name": "bucket", "id": "bucket", "editable": True},
                                        ],
                                        style_data_conditional=perc_data_bars("count %")
                                        + perc_data_bars("Event Rate")
                                        + [
                                            {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 248, 248)"},
                                            {
                                                "if": {"column_editable": True},
                                                "backgroundColor": "rgb(46,139,87)",
                                                "color": "white",
                                            },
                                            {
                                                "if": {"state": "active"},  # 'active' | 'selected'
                                                "backgroundColor": "rgba(0, 116, 217, 0.3)",
                                                "border": "1px solid rgb(0, 116, 217)",
                                            },
                                        ]
                                        + colorize_cell("bucket"),
                                        style_header={
                                            "backgroundColor": "rgb(230, 230, 230)",
                                        },
                                        editable=True,
                                    ),
                                    html.P(["Original boundaries: ", html.Code(["1,2,4"], id="original_boundaries")]),
                                    html.P(["Updated boundaries: ", html.Code(["1,2,4"], id="updated_boundaries")]),
                                ],
                                style={"padding": "0 1em 0 1em", "width": "100%"},
                            ),
                            style={"margin": "0 1em 0 1em"},
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    html.H4(children="bucketing table"),
                                    dash_table.DataTable(
                                        id="bucket_table",
                                        style_data={"whiteSpace": "normal", "height": "auto"},
                                        style_cell={
                                            "height": "auto",
                                            "overflow": "hidden",
                                            "textOverflow": "ellipsis",
                                            "maxWidth": 0,
                                            "textAlign": "center",
                                        },
                                        style_header={
                                            "backgroundColor": "rgb(230, 230, 230)",
                                        },
                                        style_data_conditional=perc_data_bars("count %")
                                        + perc_data_bars("Event Rate")
                                        + [
                                            {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 248, 248)"},
                                            {
                                                "if": {"state": "active"},  # 'active' | 'selected'
                                                "backgroundColor": "rgba(0, 116, 217, 0.3)",
                                                "border": "1px solid rgb(0, 116, 217)",
                                            },
                                        ]
                                        + colorize_cell("bucket"),
                                        style_as_list_view=True,
                                        page_size=20,
                                        columns=[
                                            {"name": "bucket", "id": "bucket"},
                                            {"name": "range", "id": "range"},
                                            {"name": "count", "id": "count"},
                                            {"name": "count %", "id": "count %"},
                                            {"name": "Non-event", "id": "Non-event"},
                                            {"name": "Event", "id": "Event"},
                                            {"name": "Event Rate", "id": "Event Rate"},
                                            {"name": "WoE", "id": "WoE"},
                                            {"name": "IV", "id": "IV"},
                                        ],
                                        editable=False,
                                    ),
                                ],
                                style={"padding": "0 1em 0 1em", "width": "100%"},
                            ),
                            style={"margin": "0 1em 0 1em"},
                        ),
                    ],
                    no_gutters=False,
                    justify="center",
                ),
            ],
            style={"margin": "1em", "padding:": "1em"},
        )

        @app.callback(
            Output("graph-prebucket", "figure"),
            [Input("input_column", "value"), Input("prebucket_table", "data")],
        )
        def plot_prebucket_bins(col, prebucket_table):
            bucket_colors = get_bucket_colors() * 4  # We repeat the colors in case there are lots of buckets
            buckets = [int(x.get("bucket")) for x in prebucket_table]
            bar_colors = [bucket_colors[i] for i in buckets]

            fig = plot_bins(self.X_prebucketed, self.y, col)
            fig.update_layout(transition_duration=50)
            fig.update_layout(showlegend=False)
            fig.update_layout(xaxis_title=col)
            fig.update_layout(title="Pre-bucketed")
            fig.update_traces(marker=dict(color=bar_colors), selector=dict(type="bar"))
            return fig

        @app.callback(
            Output("graph-bucket", "figure"),
            [Input("bucket_table", "data")],
        )
        def plot_bucket_bins(data):

            bucket_colors = get_bucket_colors() * 4  # We repeat the colors in case there are lots of buckets
            buckets = [int(x.get("bucket")) for x in data]
            bar_colors = [bucket_colors[i] for i in buckets]

            plotdf = pd.DataFrame(
                {
                    "bucket": [int(row.get("bucket")) for row in data],
                    "counts": [int(row.get("count")) for row in data],
                    "Event Rate": [row.get("Event Rate") for row in data],
                }
            )

            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            # Add traces
            fig.add_trace(
                go.Bar(x=plotdf["bucket"], y=plotdf["counts"], name="counts"),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=plotdf["bucket"], y=plotdf["Event Rate"], name="Event Rate", line=dict(color="#454c57")),
                secondary_y=True,
            )
            fig.update_layout(transition_duration=50)
            fig.update_layout(showlegend=False)
            fig.update_layout(xaxis_title="Bucket")
            # Set y-axes titles
            fig.update_yaxes(title_text="counts", secondary_y=False)
            fig.update_yaxes(title_text="event rate (%)", secondary_y=True)
            fig.update_layout(title="Bucketed")
            fig.update_xaxes(type="category")
            fig.update_traces(
                marker=dict(color=bar_colors),
                selector=dict(type="bar"),
            )
            fig.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                height=350,
            )
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
