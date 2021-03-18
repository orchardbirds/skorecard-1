from skorecard.bucketers import OptimalBucketer, DecisionTreeBucketer
from skorecard.preprocessing import WoeEncoder
from skorecard.pipeline import BucketingProcess
from skorecard.utils import NotPreBucketedError
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

import pytest


def test_bucketing_process_order(df):
    """Test that a NotPreBucketedError is raised if the bucketing pipeline is passed before the prebucketing."""
    # X = df[["LIMIT_BAL", "BILL_AMT1"]]
    # y = df["default"].values
    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    bucketing_process = BucketingProcess()
    with pytest.raises(NotPreBucketedError):
        bucketing_process.register_bucketing_pipeline(
            OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, max_n_bins=10, min_bin_size=0.05),
        )


def test_bucketing_optimization(df):
    """Test that the optimal bucketer returns less or equal number of unique buckets."""
    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    X = df[num_cols + cat_cols]
    y = df["default"].values

    bucketing_process = BucketingProcess()
    bucketing_process.register_prebucketing_pipeline(
        DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05)
    )
    bucketing_process.register_bucketing_pipeline(
        OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
        OptimalBucketer(variables=cat_cols, max_n_bins=10, min_bin_size=0.05),
    )
    X_bins = bucketing_process.fit_transform(X, y)

    for col in num_cols + cat_cols:
        assert X_bins[col].nunique() <= bucketing_process.X_prebucketed[col].nunique()


def test_bucketing_process_in_pipeline(df):
    """Test that it works fine withing a sklearn pipeline."""
    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    X = df[num_cols + cat_cols]
    y = df["default"].values

    bucketing_process = BucketingProcess()
    bucketing_process.register_prebucketing_pipeline(
        DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05)
    )
    bucketing_process.register_bucketing_pipeline(
        OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
        OptimalBucketer(variables=cat_cols, max_n_bins=10, min_bin_size=0.05),
    )

    pipeline = make_pipeline(bucketing_process, WoeEncoder(), LogisticRegression())

    pipeline.fit(X, y)
    preds = pipeline.predict_proba(X)

    assert preds.shape[0] == X.shape[0]
