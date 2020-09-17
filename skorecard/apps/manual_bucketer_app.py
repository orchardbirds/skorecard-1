import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from jupyter_dash import JupyterDash

import plotly.figure_factory as ff

import copy

# TODO make this internal to the package
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]


class ManualBucketerApp(object):
    """Dash App for manual bucketing.

    Class that contains a Dash app
    """

    def __init__(self, X, y=None, boundary_dict=None):
        """Create new dash app.

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            dash: Dash app
        """
        self.X = X
        self.y = y
        self.boundaries = boundary_dict

        app = JupyterDash(__name__, external_stylesheets=external_stylesheets)
        self.app = app

        # Add the layout
        app.layout = html.Div(
            children=[
                html.H1(children="skorecard Manual Feature Bucketing Tool"),
                html.Div(
                    children="""
                Dash: A web application framework for Python.
            """
                ),
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
                dcc.Graph(id="distr-graph"),
                dcc.Slider(min=0, max=9, marks={i: "Label {}".format(i) for i in range(10)}, value=5,),
            ]
        )

        @app.callback(Output("distr-graph", "figure"), [Input("input_column", "value")])
        def plot_dist(col):

            self.boundaries = col

            bin_size = get_bin_size(self.X, col)

            fig = ff.create_distplot([self.X[col]], [col], bin_size=bin_size)

            fig.update_layout(transition_duration=100)
            fig.update_layout(showlegend=False)
            fig.update_layout(xaxis_title=col)
            fig.update_yaxes(showticklabels=False)

            return fig

        def get_bin_size(df, col, bins=100):
            num_range = df[col].max() - df[col].min()
            return round(num_range / bins)

    def run_server(self, *args, **kwargs):
        """Start a dash server.

        Passes arguments to app.run_server()
        """
        return self.app.run_server(*args, **kwargs)

    @property
    def boundary_dict(self):
        """Ensure we return a copy of the boundary dict.

        Returns:
            dict: Boundaries per feature
        """
        return copy.deepcopy(self.boundaries)
