from skorecard import datasets
from skorecard import Skorecard
from skorecard.bucketers import DecisionTreeBucketer, OptimalBucketer
from skorecard.pipeline import BucketingProcess
import numpy as np

import pytest


@pytest.fixture()
def df():
    """Generate dataframe."""
    return datasets.load_uci_credit_card(as_frame=True)


def test_Skorecard_with_bucket_pipeline(df):
    """Test that we can use BucketingProcess to make a Skorecard() object
    """
    y = df["default"]
    X = df.drop(columns=["default"])

    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    bucketing_process = BucketingProcess(specials={'LIMIT_BAL': {'=400000.0' : [400000.0]}})
    bucketing_process.register_prebucketing_pipeline(
                                DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05),
    )
    bucketing_process.register_bucketing_pipeline(
            OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, variables_type='categorical', max_n_bins=10, min_bin_size=0.05),
    )

    model = Skorecard(bucketing_process=bucketing_process)
    model.fit(X, y)

    assert len(model.bucket_table('LIMIT_BAL')['bucket'].unique()) == 11

    X_trans = model.transform(X)
    assert all(X_trans['prediction'] < 1)
    assert all(X_trans['prediction'] > 0)


def test_Skorecard_with_bucket_process(df):
    """Test that we can use BucketingProcess to make a Skorecard() object."""
    y = df["default"]
    X = df.drop(columns=["default"])

    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    bucketing_process = BucketingProcess(specials={'LIMIT_BAL': {'=400000.0' : [400000.0]}})
    bucketing_process.register_prebucketing_pipeline(
                                DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05),
    )
    bucketing_process.register_bucketing_pipeline(
            OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
            OptimalBucketer(variables=cat_cols, variables_type='categorical', max_n_bins=10, min_bin_size=0.05),
    )

    model = Skorecard(bucketing_process=bucketing_process)
    model.fit(X, y)

    assert len(model.bucket_table('LIMIT_BAL')['bucket'].unique()) == 11

    X_trans = model.transform(X)
    assert all(X_trans['prediction'] < 1)
    assert all(X_trans['prediction'] > 0)


def test_Skorecard_with_bucket_pipeline(df):
    """Test that we can use bucketing_pipeline and prebucketing_pipeline to make a Skorecard() object."""
    y = df["default"]
    X = df.drop(columns=["default"])

    num_cols = ["LIMIT_BAL", "BILL_AMT1"]
    cat_cols = ["EDUCATION", "MARRIAGE"]

    model = Skorecard(specials={'LIMIT_BAL': {'=400000.0' : [400000.0]}},
                    prebucketing_pipeline=[DecisionTreeBucketer(variables=num_cols, max_n_bins=100, min_bin_size=0.05)],
                    bucketing_pipeline=[
                OptimalBucketer(variables=num_cols, max_n_bins=10, min_bin_size=0.05),
                OptimalBucketer(variables=cat_cols, variables_type='categorical', max_n_bins=10, min_bin_size=0.05)]

    )
    model.fit(X, y)

    assert len(model.bucket_table('LIMIT_BAL')['bucket'].unique()) == 11

    X_trans = model.transform(X)
    assert all(X_trans['prediction'] < 1)
    assert all(X_trans['prediction'] > 0)