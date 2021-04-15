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


def make_plot_figure(bucket_table: pd.DataFrame):
    """
    Make a plotly object out of a table.
    """
    if "pre-bucket" in bucket_table.columns:
        buckets = [b for b in bucket_table["pre-bucket"].values]
    else:
        buckets = [b for b in bucket_table["bucket"].values]

    plotdf = pd.DataFrame(
        {
            "bucket": buckets,
            "counts": [int(count) for count in bucket_table["Count"].values],
            "counts %": [float(count) for count in bucket_table["Count (%)"].values],
            "Event Rate": [event for event in bucket_table["Event Rate"].values],
        }
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Bar(x=plotdf["bucket"], y=plotdf["counts %"], name="Bucket count percentage"),
        secondary_y=False,
    )
    fig.update_yaxes(title_text="bucket size", secondary_y=False, tickformat=",.0%")
    fig.add_trace(
        go.Scatter(x=plotdf["bucket"], y=plotdf["Event Rate"], name="Event Rate", line=dict(color="#454c57")),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="bucket event rate", secondary_y=True, tickformat=",.0%")

    # If we have bucket information, use that to colorize the bars
    # that means a prebucket table without information on the corresponding buckets
    # wont have bars colorized.
    if "bucket" in bucket_table.columns:
        bucket_colors = get_bucket_colors() * 4  # We repeat the colors in case there are lots of buckets
        buckets = [b for b in bucket_table["bucket"].values]
        bar_colors = [bucket_colors[i] for i in buckets]
        fig.update_traces(marker=dict(color=bar_colors), selector=dict(type="bar"))

    # Other stuff
    fig.update_layout(transition_duration=50)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(type="category")
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=350,
    )
    return fig


def plot_prebucket_table(prebucket_table, column="", format=None, scale=None, width=None, height=None):
    """
    Given the prebucketed data, plot the pre-buckets.

    Args:
        prebucket_table (pd.DataFrame): the table of the prebucketed data
        X (pd.DataFrame): [description]
        y ([type], optional): [description]. Defaults to None.
        column (str): The column to plot
        format (str): The format of the image, e.g. 'png'. The default returns a plotly fig
        scale: If format is specified, the scale of the image
        width: If format is specified, the width of the image
        height: If format is specified, the image of the image

    Returns:
        fig of desired format
    """
    fig = make_plot_figure(prebucket_table)

    fig.update_layout(title=f"pre-buckets: {column}".strip())
    fig.update_layout(xaxis_title=f"{column} pre-buckets".strip())

    if format:
        img_bytes = fig.to_image(format=format, scale=scale, width=width, height=height)
        fig = Image(img_bytes)
    return fig


def plot_bucket_table(bucket_table, column="", format=None, scale=None, width=None, height=None):
    """
    Given the bucketed data, plot the buckets with Event Rate.

    Args:
        bucket_table (pd.DataFrame): the table of the bucketed data
        format (str): The format of the image, e.g. 'png'. The default returns a plotly fig
        scale: If format is specified, the scale of the image
        width: If format is specified, the width of the image
        height: If format is specified, the image of the image

    Returns:
        plotly fig
    """
    fig = make_plot_figure(bucket_table)

    fig.update_layout(title=f"buckets: {column}".strip())
    fig.update_layout(xaxis_title=f"{column} buckets".strip())

    if format is not None:
        img_bytes = fig.to_image(format=format, scale=scale, width=width, height=height)
        fig = Image(img_bytes)

    return fig
