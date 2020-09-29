from skorecard import datasets
from skorecard.bucketers import UserInputBucketer
from skorecard.bucket_mapping import BucketMapping

import pytest


@pytest.fixture()
def df():
    """Generate dataframe."""
    return datasets.load_uci_credit_card(as_frame=True)


@pytest.fixture()
def example_features_bucket_map():
    """Generate example dict."""
    return {
        "LIMIT_BAL": BucketMapping(
            feature_name="LIMIT_BAL",
            type="numerical",
            map=[10000.0, 105000.0, 265000.0, 760000.0],
            missing_bucket=None,
            right=True,
        ),
        "BILL_AMT1": BucketMapping(
            feature_name="BILL_AMT1",
            type="numerical",
            map=[-165580.0, 34211.5, 173337.5, 610723.0],
            missing_bucket=None,
            right=True,
        ),
    }


def test_manual_transformation(df, example_features_bucket_map):
    """Test that we can use an example dict for ManualBucketTransformer."""
    X = df
    y = df["default"]

    ui_bucketer = UserInputBucketer(example_features_bucket_map)
    new_X = ui_bucketer.fit_transform(X)
    assert len(new_X["LIMIT_BAL"].unique()) == 3
    assert len(new_X["BILL_AMT1"].unique()) == 3

    # specify y
    ui_bucketer = UserInputBucketer(example_features_bucket_map)
    ui_bucketer.fit(X, y)
    ui_bucketer.transform(X)
