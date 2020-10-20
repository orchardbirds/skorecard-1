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
from sklearn.pipeline import Pipeline

from skorecard.utils.exceptions import NotInstalledError
from skorecard.reporting import plot_bins

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

    def __init__(self, pipeline: Pipeline, X: pd.DataFrame, X_prebucketed: pd.DataFrame, y, index_bucket_pipeline: int):
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
        self.X_prebucketed = X_prebucketed
        self.index_bucket_pipeline = index_bucket_pipeline

        self._features_bucket_mapping = self.pipeline[index_bucket_pipeline].features_bucket_mapping_

        app = JupyterDash(__name__, external_stylesheets=external_stylesheets)
        # for url in external_stylesheets:
        #     app.css.append_css({
        #         "external_url": url
        #     })
        self.app = app

        def get_prebucket_table(col):
            vals = pd.DataFrame(self.X_prebucketed[col].value_counts()).sort_index()
            vals["bucket"] = vals.index
            vals["new_bucket"] = vals.index
            return vals

        def get_bucket_table(col):
            df = Pipeline(self.pipeline.steps[: index_bucket_pipeline + 1]).transform(self.X)
            vals = pd.DataFrame(df[col].value_counts()).sort_index()
            vals["bucket"] = vals.index
            return vals

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
                                    value=self.X.columns[0],
                                ),
                            ],
                            style={"width": "20%"},
                        )
                    )
                ),
                # dcc.Markdown(id="output-container-range-slider"),
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
                                        columns=[
                                            {"name": i, "id": i}
                                            for i in get_prebucket_table(self.X_prebucketed.columns[0])
                                        ],
                                        data=get_prebucket_table(self.X_prebucketed.columns[0]).to_dict("records"),
                                        editable=True,
                                    ),
                                ]
                            ),
                            style={"padding-left": "1em"},
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    html.P(children="bucketing table"),
                                    dash_table.DataTable(
                                        id="bucket_table",
                                        columns=[{"name": i, "id": i} for i in get_bucket_table(self.X.columns[0])],
                                        data=get_bucket_table(self.X.columns[0]).to_dict("records"),
                                    ),
                                ]
                            ),
                            style={"padding-right": "1em"},
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
            return Pipeline(self.pipeline.steps[: self.index_bucket_pipeline + 1]).transform(self.X)

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
