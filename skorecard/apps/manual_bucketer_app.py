"""Ideas for improving the app.

- Buttons for running 1d bucket transformers
- Sidebar so you can view report over all features https://dash-bootstrap-components.opensource.faculty.ai/examples/
- datatable https://dash.plotly.com/datatable
- plotly dark theme + dash dark theme? https://plotly.com/python/templates/

```python
from skorecard import datasets
from skorecard.apps import ManualBucketerApp

X, y = datasets.load_uci_credit_card(return_X_y=True)

app = ManualBucketerApp(X)
# app.run_server(mode="external")
# app.stop_server()
```
"""

import copy
import pandas as pd

from skorecard.bucket_mapping import create_bucket_feature_mapping
from skorecard.utils.exceptions import NotInstalled

# Dash + dependencies
try:
    import dash_core_components as dcc
    import dash_html_components as html
    from dash.dependencies import Input, Output
except ModuleNotFoundError:
    dcc = NotInstalled("dash_core_components", "dashboard")
    html = NotInstalled("dash_html_components", "dashboard")
    Input = NotInstalled("dash", "dashboard")
    Output = NotInstalled("dash", "dashboard")

# JupyterDash
try:
    from jupyter_dash import JupyterDash
except ModuleNotFoundError:
    JupyterDash = NotInstalled("jupyter-dash", "dashboard")

# Dash Bootstrap
try:
    import dash_bootstrap_components as dbc
except ModuleNotFoundError:
    dbc = NotInstalled("dash_bootstrap_components", "dashboard")

try:
    import plotly.figure_factory as ff
except ModuleNotFoundError:
    ff = NotInstalled("plotly", "reporting")


# TODO make this internal to the package
# external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
external_stylesheets = [dbc.themes.SKETCHY]


class ManualBucketerApp(object):
    """Dash App for manual bucketing.

    Class that contains a Dash app
    """

    def __init__(self, X, y=None, features_bucket_mapping=None):
        """Create new dash app.

        Args:
            X (pd.DataFrame): input dataframe
            y (np.array): target array
            features_bucket_mapping: Class with bucketing information for features

        Returns:
            dash: Dash app
        """
        assert isinstance(X, pd.DataFrame), "X must be pd.DataFrame"

        self.X = X
        self.y = y

        if features_bucket_mapping is None:
            # Create a basic feature mapping to get started with binning
            self._features_bucket_mapping = create_bucket_feature_mapping(self.X)
        else:
            self._features_bucket_mapping = copy.deepcopy(features_bucket_mapping)

        #         self._features_bucket_mapping[col] = bucket_mapping

        app = JupyterDash(__name__, external_stylesheets=external_stylesheets)
        self.app = app

        # Add the layout
        app.layout = html.Div(
            children=[
                html.H2(children="skorecard.ManualBucketerApp"),
                html.Div(
                    [
                        dcc.Dropdown(
                            id="input_column",
                            options=[{"label": o, "value": o} for o in self.X.columns],
                            value=self.X.columns[0],
                        ),
                    ],
                    style={"width": "20%"},
                ),
                dcc.Markdown(id="output-container-range-slider"),
                html.Div(
                    [
                        dcc.Graph(id="distr-graph"),
                        dcc.RangeSlider(
                            id="range-slider", allowCross=True, tooltip={"always_visible": True, "placement": "topLeft"}
                        ),
                    ]
                ),
            ]
        )

        @app.callback(
            [
                Output("range-slider", "min"),
                Output("range-slider", "max"),
                Output("range-slider", "value"),
                Output("range-slider", "marks"),
            ],
            [Input("input_column", "value")],
        )
        def change_col_update_slider(col):
            col_min = round(self.X[col].min(), 2)
            col_max = round(self.X[col].max(), 2)

            bucket_mapping = self._features_bucket_mapping.get(col)
            mark_edges = [round(x, 2) for x in bucket_mapping.map]

            marks = {}
            for b in mark_edges:
                marks[b] = {"label": str(b)}

            return col_min, col_max, mark_edges, marks

        @app.callback(
            Output("output-container-range-slider", "children"),
            [Input("range-slider", "value"), Input("input_column", "value")],
        )
        def update_bucket_mapping(value, col):
            return f"Boundaries for `{col}`: `{value}`"

        @app.callback(Output("distr-graph", "figure"), [Input("input_column", "value"), Input("range-slider", "value")])
        def plot_dist(col, boundaries):

            # Determine a nice bin size for histogram
            def get_bin_size(df, col, bins=100):
                num_range = df[col].max() - df[col].min()
                return round(num_range / bins)

            bin_size = get_bin_size(self.X, col)

            fig = ff.create_distplot([self.X[col]], [col], bin_size=bin_size)

            fig.update_layout(transition_duration=50)
            fig.update_layout(showlegend=False)
            fig.update_layout(xaxis_title=col)
            fig.update_yaxes(showticklabels=False)

            # Add boundary cuts as vertical lines
            if boundaries:
                shapes = []
                for b in boundaries:
                    shapes.append(dict(type="line", yref="paper", y0=0, y1=1, xref="x", x0=b, x1=b))
                fig.update_layout(shapes=shapes)

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

    @property
    def features_bucket_mapping(self):
        """
        Retrieve updated FeaturesBucketMapping from the app.

        Returns:
            dict: Boundaries per feature
        """
        return copy.deepcopy(self._features_bucket_mapping)
