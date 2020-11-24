import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from skorecard.reporting import plot_bins, bucket_table, create_report
from skorecard.apps.app_utils import determine_boundaries, get_bucket_colors
from skorecard.utils.exceptions import NotInstalledError

# Dash + dependencies
try:
    from dash.dependencies import Input, Output, State
    import dash_table
except ModuleNotFoundError:
    Input = NotInstalledError("dash", "dashboard")
    Output = NotInstalledError("dash", "dashboard")
    State = NotInstalledError("dash", "dashboard")
    dash_table = NotInstalledError("dash_table", "dashboard")

# Dash Bootstrap
try:
    import dash_bootstrap_components as dbc
except ModuleNotFoundError:
    dbc = NotInstalledError("dash_bootstrap_components", "dashboard")

# plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError:
    px = NotInstalledError("plotly", "reporting")


def add_callbacks(self):
    """
    Single place where all callbacks for the dash app are defined.
    """
    app = self.app

    @app.callback(
        Output("collapse-menu-boundaries", "is_open"),
        [Input("menu-boundaries", "n_clicks")],
        [State("collapse-menu-boundaries", "is_open")],
    )
    def toggle_collapse(n, is_open):
        """Collapse menu item.

        See https://dash-bootstrap-components.opensource.faculty.ai/docs/components/collapse/
        """
        if n:
            return not is_open
        return is_open

    @app.callback(
        Output("collapse-menu-save-versions", "is_open"),
        [Input("menu-save-versions", "n_clicks")],
        [State("collapse-menu-save-versions", "is_open")],
    )
    def toggle_collapse2(n, is_open):
        """Collapse menu item.

        See https://dash-bootstrap-components.opensource.faculty.ai/docs/components/collapse/
        """
        if n:
            return not is_open
        return is_open

    @app.callback(
        Output("collapse-menu-model-performance", "is_open"),
        [Input("menu-model-performance", "n_clicks")],
        [State("collapse-menu-model-performance", "is_open")],
    )
    def toggle_collapse3(n, is_open):
        """Collapse menu item.

        See https://dash-bootstrap-components.opensource.faculty.ai/docs/components/collapse/
        """
        if n:
            return not is_open
        return is_open

    @app.callback(
        Output("is_not_monotonic_badge", "is_open"),
        [Input("bucket_table", "data")],
    )
    def badge_is_monotonic(bucket_table):
        event_rates = [x.get("Event Rate") for x in bucket_table]
        dx = np.diff(event_rates)
        monotonic = np.all(dx <= 0) or np.all(dx >= 0)
        return not monotonic

    @app.callback(
        Output("has_5perc_badge", "is_open"),
        [Input("bucket_table", "data")],
    )
    def badge_is_has_5perc(bucket_table):
        event_perc = [x.get("count %") for x in bucket_table]
        return not all([float(x) >= 5 for x in event_perc])

    @app.callback(
        Output("original_boundaries", "children"),
        [Input("input_column", "value")],
    )
    def update_original_boundaries(col):
        return str(self.original_bucket_feature_mapping.get(col).map)

    @app.callback(
        Output("updated_boundaries", "children"), [Input("bucket_table", "data")], State("input_column", "value")
    )
    def update_updated_boundaries(bucket_table, col):
        return str(self.ui_bucketer.features_bucket_mapping.get(col).map)

    @app.callback(
        Output("input_column", "value"),
        [Input("reset-boundaries-button", "n_clicks")],
        State("input_column", "value"),
    )
    def reset_boundaries(n_clicks, col):
        original_map = self.original_bucket_feature_mapping.get(col).map
        self.ui_bucketer.features_bucket_mapping.get(col).map = original_map
        # update same column to input_colum
        # this will trigger other components to update
        return col

    @app.callback(
        Output("prebucket_table", "data"),
        [Input("input_column", "value")],
    )
    def get_prebucket_table(col):

        table = create_report(
            self.X, self.y, column=col, bucket_mapping=self.original_prebucket_feature_mapping.get(col)
        )

        # table['Count %'] = table['Count %'] * 100
        table["Event Rate"] = round(table["Event Rate"] * 100, 2)
        # table = bucket_table(x_original=self.X[col], x_bucketed=self.X_prebucketed[col], y=self.y)
        table = table.rename(columns={"bucket_id": "pre-bucket"})

        # Apply bucket mapping
        bucket_mapping = self.ui_bucketer.features_bucket_mapping.get(col)
        table["bucket"] = bucket_mapping.transform(table["pre-bucket"])
        return table.to_dict("records")

    @app.callback(
        [Output("bucket_table", "data"), Output("pre-bucket-error", "children")],
        [Input("prebucket_table", "data")],
        State("input_column", "value"),
    )
    def get_bucket_table(prebucket_table, col):

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

    @app.callback(
        Output("menu-model-performance", "children"),
        [Input("bucket_table", "data")],
    )
    def update_auc(bucket_table):
        pipe = make_pipeline(self.prebucketing_pipeline, self.ui_bucketer, self.postbucketing_pipeline)
        pipe.fit(self.X, self.y)
        yhat = [x[1] for x in pipe.predict_proba(self.X)]
        auc = roc_auc_score(self.y, yhat)
        return f"AUC: {auc:.3f}"

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
                "counts %": [float(row.get("count %")) for row in data],
                "Event Rate": [row.get("Event Rate") for row in data],
            }
        )

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Add traces
        fig.add_trace(
            go.Bar(x=plotdf["bucket"], y=plotdf["counts %"], name="counts (%)"),
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
        fig.update_yaxes(title_text="counts (%)", secondary_y=False)
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
