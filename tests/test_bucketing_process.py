from skorecard.bucketers import OptimalBucketer, DecisionTreeBucketer
from skorecard.preprocessing import WoeEncoder
from skorecard.pipeline import BucketingProcess
from skorecard.utils import NotPreBucketedError, NotBucketObjectError
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
            OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=10, min_bin_size=0.05),
        )


def test_non_bucketer_in_pipeline(df):
    """Test that putting a non-bucketer in bucket_process raises error."""
    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    bucketing_process = BucketingProcess()
    bucketing_process.register_prebucketing_pipeline(
        DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05)
    )
    with pytest.raises(NotBucketObjectError):
        bucketing_process.register_bucketing_pipeline(
            OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=10, min_bin_size=0.05),
            LogisticRegression(),  # Should break the process
        )


def test_non_bucketer_in_prebucketing_pipeline(df):
    """Test that putting a non-bucketer in pre-bucket_process raises error."""
    num_cols = ["LIMIT_BAL", "BILL_AMT1"]

    bucketing_process = BucketingProcess()
    with pytest.raises(NotBucketObjectError):
        bucketing_process.register_prebucketing_pipeline(
            DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05),
            LogisticRegression(),  # Should break the process
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
        OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=10, min_bin_size=0.05),
    )
    X_bins = bucketing_process.fit_transform(X, y)

    for col in num_cols + cat_cols:
        assert X_bins[col].nunique() <= bucketing_process.X_prebucketed[col].nunique()
        assert X_bins[col].nunique() > 1


def test_bucketing_with_specials(df):
    """Test that specials propogate."""
    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    X = df[num_cols + cat_cols]
    y = df["default"].values

    bucketing_process = BucketingProcess(specials={"LIMIT_BAL": {"=400000.0": [400000.0]}})
    bucketing_process.register_prebucketing_pipeline(
        DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05)
    )
    bucketing_process.register_bucketing_pipeline(
        OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
        OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=10, min_bin_size=0.05),
    )
    _ = bucketing_process.fit_transform(X, y)

    prebuckets = bucketing_process.prebucket_table("LIMIT_BAL")
    assert prebuckets["Count"][14] == 45.0
    assert prebuckets["label"][14] == "Special: =400000.0"

    buckets = bucketing_process.bucket_table("LIMIT_BAL")
    assert buckets["Count"][9] == 45.0
    assert buckets["label"][9] == "Special: =400000.0"


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
        OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=10, min_bin_size=0.05),
    )

    pipeline = make_pipeline(bucketing_process, WoeEncoder(), LogisticRegression())

    pipeline.fit(X, y)
    preds = pipeline.predict_proba(X)

    assert preds.shape[0] == X.shape[0]
