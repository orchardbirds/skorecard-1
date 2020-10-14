from skorecard import datasets
from skorecard.bucketers import DecisionTreeBucketer, OptimalBucketer

import pytest

from sklearn.pipeline import make_pipeline


@pytest.fixture()
def df():
    """Generate dataframe."""
    return datasets.load_uci_credit_card(as_frame=True)


def test_full_pipeline(df):
    """Tests some complete pipelines."""
    X = df.drop(columns=["default"])
    y = df["default"]

    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    prebucket_pipeline = make_pipeline(DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05))

    bucket_pipeline = make_pipeline(
        OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
        OptimalBucketer(variables=cat_cols, max_n_bins=10, min_bin_size=0.05),
    )

    pipe = make_pipeline(prebucket_pipeline, bucket_pipeline)

    pipe.fit(X, y)
    pipe.transform(X)
