from skorecard import datasets
from skorecard.bucketers import DecisionTreeBucketer
import pandas as pd
import numpy as np

import pytest


@pytest.fixture()
def df():
    """Generate dataframe."""
    return datasets.load_uci_credit_card(as_frame=True)


def test_kwargs_are_saved():
    """Test that the kwargs fed to the TreeBucketTransformer are saved."""
    tbt = DecisionTreeBucketer(variables=["LIMIT_BAL"], criterion="entropy", min_impurity_decrease=0.001,)
    assert tbt.kwargs["criterion"] == "entropy"
    assert tbt.kwargs["min_impurity_decrease"] == 0.001


def test_transform(df):
    """Test that the correct shape is returned."""
    X = df[["LIMIT_BAL", "BILL_AMT1"]]
    y = df["default"]

    tbt = DecisionTreeBucketer(variables=["LIMIT_BAL", "BILL_AMT1"], criterion="entropy", min_impurity_decrease=0.001,)
    tbt.fit(X, y)

    assert tbt.transform(X).shape == X.shape


def test_running_twice(df):
    """Test that running same bucketer twice does not change the bins.
    
    Context is that OptimalBucketer wraps optbinning.OptimalBinning, which also does prebinning using a DecisionTree.
    It's a lot more work to extract from optbinning only the optimization algorithm, so we choose to
    feed to OptimalBucketer already pre-binned buckets.
    
    We need to make sure that running it twice doesn't change the buckets.
    """
    X = df[["LIMIT_BAL", "BILL_AMT1"]]
    y = df["default"]

    tbt = DecisionTreeBucketer(variables=["LIMIT_BAL", "BILL_AMT1"])
    tbt.fit(X, y)
    X_prebucketed = tbt.transform(X)

    tbt2 = DecisionTreeBucketer(variables=["LIMIT_BAL", "BILL_AMT1"])
    tbt2.fit(X_prebucketed, y)
    X_bucketed = tbt2.transform(X_prebucketed)

    assert X_prebucketed["BILL_AMT1"].value_counts().equals(X_bucketed["BILL_AMT1"].value_counts())

    # But what about using a different random_state?
    tbt3 = DecisionTreeBucketer(variables=["LIMIT_BAL", "BILL_AMT1"], random_state=1)
    tbt3.fit(X_prebucketed, y)
    X_bucketed = tbt3.transform(X_prebucketed)

    assert X_prebucketed["BILL_AMT1"].value_counts().equals(X_bucketed["BILL_AMT1"].value_counts())


def test_specials_filters(df):
    """Test that when adding specials,the binner performs as expected.

    Context: special values should be binned in their own bin.
    """
    X = df[["LIMIT_BAL", "BILL_AMT1"]]
    y = df["default"]

    specials = {"LIMIT_BAL": {"=50000": [50000], "in [20001,30000]": [20000, 30000]}}

    f = DecisionTreeBucketer._filter_specials_for_fit

    X_flt, y_flt = f(X["LIMIT_BAL"], y, specials=specials["LIMIT_BAL"])

    assert X_flt[X_flt.isin([20000, 30000, 50000])].shape[0] == 0
    assert y_flt.equals(y[~X["LIMIT_BAL"].isin([20000, 30000, 50000])])


def test_all_data_is_special(df):
    """Test that when all the data is in the special buckets, the code does not crash."""
    X = df[["LIMIT_BAL", "BILL_AMT1"]]
    y = df["default"]

    X["all_zeros"] = pd.Series([999 for i in range(X.shape[0])])
    X["all_nans"] = pd.Series([np.nan for i in range(X.shape[0])])
    specials = {"all_zeros": {"=999": [999]}}

    tbt = DecisionTreeBucketer(variables=["all_zeros", "all_nans"], specials=specials)
    tbt.fit(X, y)
    X_prebucketed = tbt.transform(X)
    # Bucket 0 will still be a potential value (between -inf and +inf)
    # the next bucket (bucket 1) is reserved for nans,
    # therefore the special bucket starts at index 2.
    assert X_prebucketed["all_nans"].unique() == 1
    assert X_prebucketed["all_zeros"].unique() == 2


def test_missing_default(df_with_missings) -> None:
    """Test that missing values are assigned to the right bucket."""
    X = df_with_missings
    y = df_with_missings["default"].values

    BUCK = DecisionTreeBucketer(variables=["LIMIT_BAL"], random_state=1)
    BUCK.fit(X, y)
    X['LIMIT_BAL_trans'] = BUCK.transform(X[['LIMIT_BAL']])

    missing_bucket = [f for f in BUCK.features_bucket_mapping_['LIMIT_BAL'].labels.keys()][-1]
    assert BUCK.features_bucket_mapping_['LIMIT_BAL'].labels[missing_bucket] == 'Missing'
    assert X[np.isnan(X['LIMIT_BAL'])].shape[0] == X[X['LIMIT_BAL_trans'] == missing_bucket].shape[0]


def test_missing_manual(df_with_missings) -> None:
    """Test that missing values are assigned to the right bucket."""
    X = df_with_missings
    y = df_with_missings["default"].values

    bucketer = DecisionTreeBucketer(variables=["LIMIT_BAL"], random_state=1, missing_treatment={'LIMIT_BAL': 0})
    bucketer.fit(X, y)
    X['LIMIT_BAL_trans'] = bucketer.transform(X[['LIMIT_BAL']])

    assert X[np.isnan(X['LIMIT_BAL_trans'])]['LIMIT_BAL'].sum() == 0