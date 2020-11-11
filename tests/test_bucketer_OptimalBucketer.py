import pytest
import numpy as np
import pandas as pd
from optbinning import OptimalBinning
from sklearn.pipeline import make_pipeline

from skorecard.bucketers import OptimalBucketer, DecisionTreeBucketer
from skorecard.bucket_mapping import BucketMapping
from skorecard.pipeline import make_coarse_classing_pipeline
from skorecard.utils import NotPreBucketedError


def test_optimal_binning_prebinning(df):
    """Ensure we have prevented prebinning correctly.
    
    optbinning.OptimalBinning() does pre-binning by default.
    In skorecard, we want more control, so force the user to do this explicitly.
    
    Extracting only the optimizer from optbinning is quite involved.
    Instead, we'll let the prebinning (a DecisionTreeClassifier) use our user defined splits.
    These splits are simply the unique values from the prebucketed feature.
    
    This tests checks the output is not changed.
    """
    X = df[["LIMIT_BAL", "BILL_AMT1"]]
    y = df["default"].values

    # optbinning saves the user_defined_splits for prebinning
    # https://github.com/guillermo-navas-palencia/optbinning/blob/396b9bed97581094167c9eb4744c2fd1fb5c7408/optbinning/binning/binning.py#L741
    # then applies np.digitize(right=False)
    # https://github.com/guillermo-navas-palencia/optbinning/blob/396b9bed97581094167c9eb4744c2fd1fb5c7408/optbinning/binning/binning.py#L1048
    # If these are the prebuckted values
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # Internally this happens:
    binned_x = np.digitize(x=x, bins=x, right=False)
    # which means a shift by 1
    assert all(binned_x == x + 1)
    # which means internally it is shifted by 1, but that doesn't really matter,
    # as tests below show, because still same unique buckets.

    # transform a feature using new flow
    binner = OptimalBinning(
        name="BILL_AMT1",
        dtype="numerical",
        solver="cp",
        monotonic_trend="auto_asc_desc",
        min_prebin_size=0.05,
        max_n_prebins=20,
        min_bin_size=0.05,
        max_n_bins=10,
        cat_cutoff=0.05,
        time_limit=25,
    )
    old = binner.fit_transform(X["BILL_AMT1"], y, metric="indices")

    pipe = make_pipeline(
        DecisionTreeBucketer(variables=["BILL_AMT1"], max_n_bins=20, min_bin_size=0.05),
        make_coarse_classing_pipeline(OptimalBucketer(variables=["BILL_AMT1"], max_n_bins=10, min_bin_size=0.05),),
    )
    new = pipe.fit_transform(X, y)["BILL_AMT1"]

    assert all(np.equal(old, new.values))

    # now for categoricals


def test_optimal_binning_numerical(df):
    """Tests the wrapper of optbinning.OptimalBinning()."""
    X = df[["LIMIT_BAL", "BILL_AMT1"]]
    y = df["default"].values

    X_prebucketed = DecisionTreeBucketer(max_n_bins=20, min_bin_size=0.05).fit_transform(X, y)

    obt = OptimalBucketer(variables=["LIMIT_BAL", "BILL_AMT1"])
    with pytest.raises(NotPreBucketedError):
        obt.fit(X, y)
    obt.fit(X_prebucketed, y)
    X_trans = obt.transform(X_prebucketed)
    assert len(X_trans["LIMIT_BAL"].unique()) == 9
    assert len(X_trans["BILL_AMT1"].unique()) == 6

    obt = OptimalBucketer(variables=["LIMIT_BAL"])
    obt.fit(X_prebucketed, y)
    X_trans = obt.transform(X_prebucketed)
    assert len(X_trans["LIMIT_BAL"].unique()) == 9

    # Test the transforms work well
    optb = OptimalBinning(name="LIMIT_BAL", dtype="numerical", solver="cp", max_n_prebins=100, max_n_bins=10)
    optb.fit(X_prebucketed["LIMIT_BAL"], y)
    ref = optb.transform(X_prebucketed["LIMIT_BAL"], metric="indices")
    skore = X_trans["LIMIT_BAL"]
    assert len(np.unique(ref)) == len(np.unique(skore))

    # Multiple columns in a df, should keep transformation equal
    df1 = obt.transform(pd.DataFrame([0, 30_000], columns=["LIMIT_BAL"]))
    df2 = obt.transform(pd.DataFrame([[0, 0], [30_000, 30_000]], columns=["LIMIT_BAL", "BILL_AMT1"]))
    assert df1["LIMIT_BAL"].equals(df2["LIMIT_BAL"])

    # Note our bins were 1-indexed in the past
    # This unit test is there to make sure our bins are zero-indexed
    obt.fit(X, y)
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
        feature_name="EDUCATION",
        type="categorical",
        map={1: 0, 3: 1, 2: 2, 5: 3, 4: 3, 6: 3, 0: 3},
        right=False,
        labels=[[1], [3], [2], [0, 4, 5, 6], "other"],
    )

    optb = OptimalBinning(
        name="EDUCATION", dtype="categorical", solver="cp", cat_cutoff=0.05, max_n_prebins=100, max_n_bins=10
    )
    optb.fit(X["EDUCATION"], y)
    ref = optb.transform(X["EDUCATION"], metric="indices")
    X_trans["EDUCATION"].equals(pd.Series(ref))

    raise NotImplementedError("implement a test with specials in categorical binning")
