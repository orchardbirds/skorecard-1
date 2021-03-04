from skorecard.bucketers import OptimalBucketer  # , DecisionTreeBucketer
from skorecard.pipeline import BucketingProcess
from skorecard.utils import NotPreBucketedError

import pytest


def test_bucketing_process_order(df):
    """Test that a NotPreBucketedError is raised if the bucketing pipeline is passed before the prebucketing."""
    # X = df[["LIMIT_BAL", "BILL_AMT1"]]
    # y = df["default"].values
    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    bucketing_process = BucketingProcess(specials={"column": {"label": "value"}})
    with pytest.raises(NotPreBucketedError):
        bucketing_process.register_bucketing_pipeline(
            OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, max_n_bins=10, min_bin_size=0.05),
        )
