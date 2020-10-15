import pytest
import numpy as np
import pandas as pd
from optbinning import OptimalBinning

from skorecard import datasets
from skorecard.bucketers import OptimalBucketer
from skorecard.bucket_mapping import BucketMapping


@pytest.fixture()
def df():
    """Generate dataframe."""
    return datasets.load_uci_credit_card(as_frame=True)


def test_optimal_binning_numerical(df):
    """Tests the wrapper of optbinning.OptimalBinning()."""
    X = df[["LIMIT_BAL", "BILL_AMT1"]]
    y = df["default"].values

    obt = OptimalBucketer(variables=["LIMIT_BAL", "BILL_AMT1"])
    obt.fit(X, y)
    X_trans = obt.transform(X)
    assert len(X_trans["LIMIT_BAL"].unique()) == 9
    assert len(X_trans["BILL_AMT1"].unique()) == 6

    obt = OptimalBucketer(variables=["LIMIT_BAL"])
    obt.fit(X, y)
    X_trans = obt.transform(X)
    assert len(X_trans["LIMIT_BAL"].unique()) == 9

    # Test the transforms work well
    optb = OptimalBinning(name="LIMIT_BAL", dtype="numerical", solver="cp", max_n_prebins=100, max_n_bins=10)
    optb.fit(X["LIMIT_BAL"], y)
    ref = optb.transform(X["LIMIT_BAL"], metric="indices")
    skore = X_trans["LIMIT_BAL"]
    assert len(np.unique(ref)) == len(np.unique(skore))

    ref[:10]
    skore[:10]
    optb.transform([0, 30_000], metric="indices")

    # Multiple columns in a df, should keeep transformation equal
    df1 = obt.transform(pd.DataFrame([0, 30_000], columns=["LIMIT_BAL"]))
    df2 = obt.transform(pd.DataFrame([[0, 0], [30_000, 30_000]], columns=["LIMIT_BAL", "BILL_AMT1"]))
    assert df1["LIMIT_BAL"].equals(df2["LIMIT_BAL"])

    # Note our bins were 1-indexed in the past
    # This unit test is there to make sure our bins are zero-indexed
    assert all(obt.features_bucket_mapping_.get("LIMIT_BAL").transform([0, 30_000]) == np.array([0, 1]))

    # optb.binning_table.build()
    # optb.splits
    # obt.features_bucket_mapping_.get('LIMIT_BAL')


def test_optimal_binning_categorical(df):
    """Test categoricals."""
    X = df[["LIMIT_BAL", "BILL_AMT1", "EDUCATION"]]
    y = df["default"].values

    obt = OptimalBucketer(variables=["EDUCATION"], variables_type="categorical")
    obt.fit(X, y)
    X_trans = obt.transform(X)
    assert len(X_trans["EDUCATION"].unique()) == 4

    assert obt.features_bucket_mapping_.get("EDUCATION") == BucketMapping(
        feature_name="EDUCATION", type="categorical", map={1: 0, 3: 1, 2: 2, 5: 3, 4: 3, 6: 3, 0: 3}, right=False,
    )

    optb = OptimalBinning(
        name="EDUCATION", dtype="categorical", solver="cp", cat_cutoff=0.05, max_n_prebins=100, max_n_bins=10
    )
    optb.fit(X["EDUCATION"], y)
    ref = optb.transform(X["EDUCATION"], metric="indices")
    X_trans["EDUCATION"].equals(pd.Series(ref))
