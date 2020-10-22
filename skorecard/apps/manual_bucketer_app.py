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

import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline

from skorecard.utils.exceptions import NotInstalledError
from skorecard.reporting import plot_bins, bucket_table
from skorecard.pipeline import split_pipeline

# Dash + dependencies
try:
    import dash_core_components as dcc
    import dash_html_components as html
    from dash.dependencies import Input, Output
    import dash_table
except ModuleNotFoundError:
    dcc = NotInstalledError("dash_core_components", "dashboard")
    html = NotInstalledError("dash_html_components", "dashboard")
    Input = NotInstalledError("dash", "dashboard")
    Output = NotInstalledError("dash", "dashboard")
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

        app = JupyterDash(__name__, external_stylesheets=external_stylesheets)
        self.app = app

        @app.callback(
            Output("prebucket_table", "data"),
            [Input("input_column", "value")],
        )
        def get_prebucket_table(col):
            table = bucket_table(x_original=self.X[col], x_bucketed=self.X_prebucketed[col], y=self.y)
            table = table.rename(columns={"bucket": "pre-bucket"})
            return table.to_dict("records")

        @app.callback(
            Output("bucket_table", "data"),
            [Input("input_column", "value")],
        )
        def get_bucket_table(col):
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
                                    html.P(children="pre-bucketing table"),
                                    dash_table.DataTable(
                                        id="prebucket_table",
                                        style_data={
                                            "whiteSpace": "normal",
                                            "height": "auto",
                                        },
                                        style_cell={"overflow": "hidden", "textOverflow": "ellipsis", "maxWidth": 0},
                                        style_as_list_view=True,
                                        page_size=20,
                                        columns=[
                                            {"name": "pre-bucket", "id": "pre-bucket"},
                                            {"name": "min", "id": "min"},
                                            {"name": "max", "id": "max"},
                                            {"name": "count", "id": "count"},
                                        ],
                                        editable=True,
                                    ),
                                ],
                                style={"padding-left": "1em", "width": "400px"},
                            ),
                            style={"margin-left": "1em"},
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    html.P(children="bucketing table"),
                                    dash_table.DataTable(
                                        id="bucket_table",
                                        style_data={
                                            "whiteSpace": "normal",
                                            "height": "auto",
                                        },
                                        style_cell={"overflow": "hidden", "textOverflow": "ellipsis", "maxWidth": 0},
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

        # @app.callback(
        #     [
        #         Output("range-slider", "min"),
        #         Output("range-slider", "max"),
        #         Output("range-slider", "value"),
        #         Output("range-slider", "marks"),
        #     ],
        #     [Input("input_column", "value")],
        # )
        # def change_col_update_slider(col):
        #     col_min = round(self.X_prebucketed[col].min(), 2)
        #     col_max = round(self.X_prebucketed[col].max(), 2)

        #     bucket_mapping = self._features_bucket_mapping.get(col)
        #     mark_edges = [round(x, 2) for x in bucket_mapping.map]

        #     marks = {}
        #     for b in mark_edges:
        #         marks[b] = {"label": str(b)}

        #     return col_min, col_max, mark_edges, marks

        # @app.callback(
        #     Output("output-container-range-slider", "children"),
        #     [Input("range-slider", "value"), Input("input_column", "value")],
        # )
        # def update_bucket_mapping(value, col):
        #     return f"Boundaries for `{col}`: `{value}`"

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
            [Input("input_column", "value")],
        )
        def plot_dist2(col):
            fig = plot_bins(get_bucketed_X(), col)
            fig.update_layout(transition_duration=50)
            fig.update_layout(showlegend=False)
            fig.update_layout(xaxis_title=col)
            fig.update_layout(title="Bucketed")
            return fig

        def get_bucketed_X():
            return make_pipeline(self.prebucketing_pipeline, self.ui_bucketer).transform(self.X)

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
