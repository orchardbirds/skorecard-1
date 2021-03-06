from skorecard.bucketers import OptimalBucketer, DecisionTreeBucketer
from skorecard.preprocessing import WoeEncoder
from skorecard.pipeline import BucketingProcess
from skorecard.pipeline.bucketing_process import find_remapped_specials
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

    X_prebucketed = bucketing_process.prebucketing_pipeline.transform(X)
    for col in num_cols + cat_cols:
        assert X_bins[col].nunique() <= X_prebucketed[col].nunique()
        assert X_bins[col].nunique() > 1


def test_bucketing_with_specials(df):
    """Test that specials propogate."""
    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    X = df[num_cols + cat_cols]
    y = df["default"].values

    the_specials = {"LIMIT_BAL": {"=400000.0": [400000.0]}}
    bucketing_process = BucketingProcess(specials=the_specials)
    bucketing_process.register_prebucketing_pipeline(
        DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05),
    )
    bucketing_process.register_bucketing_pipeline(
        OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
        OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=10, min_bin_size=0.05),
    )
    _ = bucketing_process.fit_transform(X, y)

    # Make sure all the prebucketers have the specials assigned
    for step in bucketing_process.prebucketing_pipeline:
        assert step.specials == the_specials

    prebuckets = bucketing_process.prebucket_table("LIMIT_BAL")
    assert prebuckets["Count"][14] == 45.0
    assert prebuckets["label"][14] == "Special: =400000.0"

    buckets = bucketing_process.bucket_table("LIMIT_BAL")
    assert buckets["Count"][10] == 45.0
    assert buckets["label"][10] == "Special: =400000.0"


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


def test_bucketing_process_with_numerical_specials(df):
    """
    Test we get expected results for numerical specials.
    """
    y = df["default"]
    X = df.drop(columns=["default"])

    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    bucketing_process = BucketingProcess(specials={"LIMIT_BAL": {"=400000.0": [400000.0]}})
    bucketing_process.register_prebucketing_pipeline(
        DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05),
        DecisionTreeBucketer(variables=cat_cols, max_n_bins=100, min_bin_size=0.05),
    )
    bucketing_process.register_bucketing_pipeline(
        OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
        OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=10, min_bin_size=0.05),
    )

    bucketing_process.fit(X, y)

    table = bucketing_process.prebucket_table("LIMIT_BAL")
    assert len(table["bucket"].unique()) == 10
    assert table[["label"]].values[-1] == "Special: =400000.0"

    table = bucketing_process.prebucket_table("MARRIAGE")
    assert table.shape[0] == 3


def test_bucketing_process_with_categorical_specials(df):
    """
    Test we get expected results for numerical specials.
    """
    y = df["default"]
    X = df.drop(columns=["default"])

    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    bucketing_process = BucketingProcess(specials={"MARRIAGE": {"=0": [0]}})
    bucketing_process.register_prebucketing_pipeline(
        DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05),
        DecisionTreeBucketer(variables=cat_cols, max_n_bins=100, min_bin_size=0.05),
    )
    bucketing_process.register_bucketing_pipeline(
        OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
        OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=10, min_bin_size=0.05),
    )

    bucketing_process.fit(X, y)

    table = bucketing_process.prebucket_table("MARRIAGE")
    assert table.shape[0] == 4
    assert table["label"][3] == "Special: =0"

    def test_bucketing_process_summary(df):
        """
        Test bucketing process.

        Test we get expected results for .summary()
        """
        y = df["default"]
        X = df.drop(columns=["default"])

        num_cols = ["LIMIT_BAL", "BILL_AMT1"]
        cat_cols = ["EDUCATION", "MARRIAGE"]

        bucketing_process = BucketingProcess(specials={"MARRIAGE": {"=0": [0]}})
        bucketing_process.register_prebucketing_pipeline(
            DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05),
            DecisionTreeBucketer(variables=cat_cols, max_n_bins=100, min_bin_size=0.05),
        )
        bucketing_process.register_bucketing_pipeline(
            OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, variables_type="categorical", max_n_bins=10, min_bin_size=0.05),
        )

        bucketing_process.fit(X, y)
        table = bucketing_process.summary()
        assert set(table.columns) == set(["column", "num_prebuckets", "num_buckets", "dtype"])
        assert table[table["column"] == "pet_ownership"]["num_prebuckets"].values[0] == "not_bucketed"
        assert table[table["column"] == "pet_ownership"]["num_buckets"].values[0] == "not_bucketed"
        assert len(table["dtype"].unique()) == 3


def test_remapping_specials():
    """
    Test remapping works.
    """
    bucket_labels = {
        0: "(-inf, 25000.0)",
        1: "[25000.0, 45000.0)",
        2: "[45000.0, 55000.0)",
        3: "[55000.0, 75000.0)",
        4: "[75000.0, 85000.0)",
        5: "[85000.0, 105000.0)",
        6: "[105000.0, 145000.0)",
        7: "[145000.0, 175000.0)",
        8: "[175000.0, 225000.0)",
        9: "[225000.0, 275000.0)",
        10: "[275000.0, 325000.0)",
        11: "[325000.0, 385000.0)",
        12: "[385000.0, inf)",
        13: "Missing",
        14: "Special: =400000.0",
    }

    var_specials = {"=400000.0": [400000.0]}

    assert find_remapped_specials(bucket_labels, var_specials) == {"=400000.0": [14]}

    assert find_remapped_specials(bucket_labels, None) == {}
    assert find_remapped_specials(None, None) == {}

    bucket_labels = {13: "Special: =12345 or 123456", 14: "Special: =400000.0"}

    var_specials = {"=400000.0": [400000.0], "=12345 or 123456": [12345, 123456]}
    assert find_remapped_specials(bucket_labels, var_specials) == {"=400000.0": [14], "=12345 or 123456": [13]}
