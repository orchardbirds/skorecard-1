from skorecard.reporting import create_report
from skorecard import datasets
from skorecard.bucketers import EqualWidthBucketer, OrdinalCategoricalBucketer

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from dabl import detect_types

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import streamlit as st


def make_example_pipeline(X, n_bins):
    """Create an example pipeline.

    It contains an EqualWidthBucketer and an Ordinal Categorical Bucketer.
    Number of bins must be given.

    Args:
        X (np.array): training data
        n_bins (int): the number of bins for the bins

    Returns:
        bucket_pipeline: The sklearn pipeline of bucketers only
        pipeline: The full sklearn pipeline
    """
    n_bins = int(n_bins)
    detected_types = detect_types(X)
    cat_columns = X.columns[(detected_types["categorical"]) | (detected_types["low_card_int"])]
    num_columns = X.columns[(detected_types["continuous"]) | (detected_types["dirty_float"])]
    bucket_pipeline = make_pipeline(
        EqualWidthBucketer(bins=n_bins, variables=list(num_columns)),
        OrdinalCategoricalBucketer(variables=list(cat_columns)),
    )

    pipeline = Pipeline(
        [("bucketing", bucket_pipeline), ("one-hot-encoding", OneHotEncoder()), ("lr", LogisticRegression())]
    )

    return bucket_pipeline, pipeline


def generate_bucketed_statistics(X, y, n_bins, column, method):
    """Calculates the AUC of the pipeline based on the number of bins.

    Also generates the bucketing report for the specified column.

    Args:
        X (np.array): Training data
        y (np.array): Training labels
        n_bins (int): Number of bins
        column (str): Name of the column for which the report is generated
        method (str): Name of bucket to use

    Returns:
        df: The generated report
        auc: The AUC produced by the number of bins
    """
    bucket_pipeline, pipeline = make_example_pipeline(X, n_bins)

    pipeline.fit(X, y)
    auc = f"AUC = {roc_auc_score(y, pipeline.predict_proba(X)[:, 1]):.4f}"

    if method == "categorical":
        bucketer = bucket_pipeline.named_steps["ordinalcategoricalbucketer"]
        # todo: fix streamlit visuals of cat bucket.
    elif method == "equal_width":
        bucketer = bucket_pipeline.named_steps["equalwidthbucketer"]

    df = create_report(X, y, column, bucketer)

    return df, auc


def create_barplot(df, auc):
    """Creates a plotly barplot for the percentage of instances per bucket.

    Also plots a lineplot for the default rates per bucket.

    Args:
        df: Report dataframe
        auc (str): The AUC of the pipeline

    Returns:
        fig: The full plotly figure
    """
    bin_number = df.shape[0]
    df = df.sort_values("BUCKET")
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # fig.update_traces(mode='lines+markers')
    fig.update_traces()

    fig.update_xaxes(showgrid=False)

    fig.update_yaxes(showgrid=False)

    fig.add_annotation(
        x=0,
        y=0.85,
        xanchor="left",
        yanchor="bottom",
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        bgcolor="rgba(255, 255, 255, 0.5)",
        text=bin_number,
    )

    # Add traces
    fig.add_trace(go.Bar(x=df["BUCKET"], y=df["PERCENTAGE_IN_BUCKET"], name="Percentages"), secondary_y=False)

    fig.add_trace(go.Scatter(x=df["BUCKET"], y=df["DEFAULT_RATE"], name="Default Rates"), secondary_y=True)
    fig.update_yaxes(title_text="Percentage", secondary_y=False)
    fig.update_yaxes(title_text="Default Rate", secondary_y=True)

    fig.update_layout(title=auc, xaxis_title="Bucket Number", font_family="Courier New")
    return fig


def generate_streamlit_visuals(X, y, n_bins, column, method):
    """Gathers the bucketed statistics.

    Also plots the figure and report dataframe in streamlit.

    Args:
        X (np.array): Training data
        y (np.array): Training labels
        n_bins (int): Number of buckets for bucketers
        column (str): Name of column for which to generate the report
        method (str): Name of method to visualise

    Returns:
        The figure.
    """
    df, auc = generate_bucketed_statistics(X, y, n_bins, column, method)
    st.dataframe(df.head(n_bins))
    return create_barplot(df, auc)


def create_streamlit_example():
    """Run full flow.

    Run this file with 'streamlit run example_streamlit.py'
    """
    st.title("Example Bucketing Report")
    n_bins = st.slider("n_bins", 1, 50)
    X, y = datasets.load_uci_credit_card(return_X_y=True)
    columns = st.selectbox("Feature", (X.columns))
    methods = st.selectbox("Bucketer", ("categorical", "equal_width"))
    st.write(generate_streamlit_visuals(X, y, n_bins=n_bins, column=columns, method=methods))
    return None


if __name__ == "__main__":
    create_streamlit_example()
