import pandas as pd

from skorecard.utils.exceptions import NotInstalledError

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError:
    px = NotInstalledError("plotly", "reporting")


def plot_bins(X, y, col):
    """Plot bin table."""
    assert isinstance(X, pd.DataFrame)

    # Create plotting df
    val_counts = X[col].value_counts()
    plotdf = pd.DataFrame({"bucket": val_counts.index, "counts": val_counts.values})

    # Add event rates
    ref = pd.DataFrame()
    ref["y"] = y
    ref["bucket"] = X[col]
    er = ref.groupby(["bucket", "y"]).agg({"y": ["count"]}).reset_index()
    er.columns = [" ".join(col).strip() for col in er.columns.values]
    er = er.pivot(index="bucket", columns="y", values="y count").fillna(0)
    er = er.rename(columns={0: "Non-event", 1: "Event"})
    er["Event Rate"] = round((er["Event"] / (er["Event"] + er["Non-event"])) * 100, 2).astype(str) + "%"
    plotdf = plotdf.merge(er, how="left", on="bucket").sort_values("bucket")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Bar(x=plotdf["bucket"], y=plotdf["counts"], name="Bucket counts"),
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
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=350,
    )
    return fig
