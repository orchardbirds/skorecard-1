import pandas as pd

from skorecard.utils.exceptions import NotInstalledError

try:
    import plotly.express as px
except ModuleNotFoundError:
    px = NotInstalledError("plotly", "reporting")


def plot_bins(X, col):
    """Plot bin table."""
    assert isinstance(X, pd.DataFrame)

    val_counts = X[col].value_counts()

    plotdf = pd.DataFrame({col: val_counts.index, "counts": val_counts.values})

    fig = px.bar(plotdf, x=col, y="counts")
    return fig
