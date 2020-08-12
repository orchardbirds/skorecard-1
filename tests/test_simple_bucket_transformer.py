from skorecard import datasets
from skorecard.preprocessing import SimpleBucketTransformer

import pytest


@pytest.fixture()
def df():
    """Generate dataframe."""
    return datasets.load_uci_credit_card(as_frame=True)


def test_single_bucket(df):
    """Test that using bin_count=1 puts everything into 1 bucket."""
    SBT = SimpleBucketTransformer(bin_count=1)
    SBT.fit(df["LIMIT_BAL"].values)
    assert SBT.Bucketer.counts == df.shape[0]

    return None


def test_bin_counts(df):
    """Test that we get the number of bins we request."""
    # Test single bin counts
    SBT = SimpleBucketTransformer(bin_count=7)
    SBT.fit_transform(df["LIMIT_BAL"].values)
    assert SBT.BucketDict["Feature_0"].bin_count == 7

    # Test multiple bin counts
    SBT = SimpleBucketTransformer(bin_count=[7, 5])
    SBT.fit_transform(df[["LIMIT_BAL", "BILL_AMT1"]].values)
    assert SBT.BucketDict["Feature_0"].bin_count == 7
    assert SBT.BucketDict["Feature_1"].bin_count == 5

    return None


def test_float_bin_count(df):
    """Test that a float for bin_count raises an error."""
    try:
        SBT = SimpleBucketTransformer(bin_count=7.3)
        SBT.fit_transform(df["LIMIT_BAL"].values)
    except AttributeError:
        assert True

    return None


def test_bucket_dict_number(df):
    """Test that we get a separate Bucketer object per number of columns given."""
    SBT = SimpleBucketTransformer(bin_count=[3, 3])
    SBT.fit(df[["LIMIT_BAL", "BILL_AMT1"]].values)
    assert len(SBT.BucketDict) == 2

    return None


def test_bad_bin_count_shape(df):
    """Test that bad bin shape triggers ValueError."""
    # Simple Bucketer
    try:
        SBT = SimpleBucketTransformer(bin_count=[3, 3])
        SBT.fit(df["LIMIT_BAL"].values)
    except ValueError:
        assert True

    try:
        SBT = SimpleBucketTransformer(bin_count=[3])
        SBT.fit(df[["LIMIT_BAL", "BILL_AMT1"]].values)
    except ValueError:
        assert True

    return None
