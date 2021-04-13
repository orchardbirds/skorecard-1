import pandas as pd
from skorecard.apps.app_utils import get_bucket_colors

from skorecard.utils.exceptions import NotInstalledError

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError:
    px = NotInstalledError("plotly", "reporting")

try:
    from IPython.display import Image
except ModuleNotFoundError:
    Image = NotInstalledError("psutil")

def plot_bins(X, y, col):
    """Plot bin table."""
    assert isinstance(X, pd.DataFrame)

    # Create plotting df
    val_counts = X[col].value_counts()
    val_counts_normalized = X[col].value_counts(normalize=True)
    plotdf = pd.DataFrame(
        {"bucket": val_counts.index, "counts": val_counts.values, "counts %": val_counts_normalized.values}
    )

    # Add event rates
    ref = pd.DataFrame()
    ref["y"] = y
    ref["bucket"] = X[col]
    er = ref.groupby(["bucket", "y"]).agg({"y": ["count"]}).reset_index()
    er.columns = [" ".join(col).strip() for col in er.columns.values]
    er = er.pivot(index="bucket", columns="y", values="y count").fillna(0)
    er = er.rename(columns={0: "Non-event", 1: "Event"})
    er["Event Rate"] = round((er["Event"] / (er["Event"] + er["Non-event"])) * 100, 2)
    plotdf = plotdf.merge(er, how="left", on="bucket").sort_values("bucket")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces
    fig.add_trace(
        go.Bar(x=plotdf["bucket"], y=plotdf["counts %"] * 100, name="Bucket count percentage"),
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
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=350,
    )
    return fig

def plot_prebucket_table(prebucket_table, X, y, column, format=None):
    """
    Given the prebucketed data, plot the pre-buckets

    Args:
        prebucket_table (pd.DataFrame): the table of the prebucketed data
        X (pd.DataFrame): [description]
        y ([type], optional): [description]. Defaults to None.
        column (str): The column to plot
        format (str): The format of the image, e.g. 'png'. The default returns a plotly fig

    Returns:
        fig of desired format
    """
    bucket_colors = get_bucket_colors() * 4  # We repeat the colors in case there are lots of buckets
    buckets = [prebucket for prebucket in prebucket_table['pre-bucket'].values]
    bar_colors = [bucket_colors[i] for i in buckets]

    fig = plot_bins(X, y, column)
    fig.update_layout(transition_duration=50)
    fig.update_layout(showlegend=False)
    fig.update_layout(xaxis_title=column)
    fig.update_layout(title="Pre-bucketed")
    fig.update_traces(marker=dict(color=bar_colors), selector=dict(type="bar"))

    if format is not None:
        img_bytes = fig.to_image(format=format)
        fig = Image(img_bytes)
    return fig

def plot_bucket_table(bucket_table, format=None):
    """
    Given the bucketed data, plot the buckets with Event Rate

    Args:
        bucket_table (pd.DataFrame): the table of the bucketed data
        format (str): The format of the image, e.g. 'png'. The default returns a plotly fig

    Returns:
        plotly fig
    """
    bucket_colors = get_bucket_colors() * 4  # We repeat the colors in case there are lots of buckets
    buckets = [int(bucket) for bucket in bucket_table['bucket'].values]
    bar_colors = [bucket_colors[i] for i in buckets]

    plotdf = pd.DataFrame(
        {
            "bucket": buckets,
            "counts": [int(count) for count in bucket_table['Count'].values],
            "counts %": [float(count) for count in bucket_table['Count (%)'].values],
            "Event Rate": [event for event in bucket_table['Event Rate'].values],
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

    if format is not None:
        img_bytes = fig.to_image(format=format)
        fig = Image(img_bytes)

    return fig
