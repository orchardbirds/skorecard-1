from skorecard import datasets
from skorecard.preprocessing import SimpleBucketTransformer
from skorecard.utils.exceptions import DimensionalityError


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
    with pytest.raises(DimensionalityError):
        SBT.fit_transform(df[["LIMIT_BAL", "BILL_AMT1"]].values)


def test_float_bin_count(df):
    """Test that a float for bin_count raises an error."""
    with pytest.raises(AttributeError):
        SBT = SimpleBucketTransformer(bin_count=7.3)
        SBT.fit_transform(df["LIMIT_BAL"].values)


def test_bucket_dict_number(df):
    """Test that we get an exception if the wrong bucketer is raised."""
    SBT = SimpleBucketTransformer(bin_count=[3, 3])
    with pytest.raises(DimensionalityError):
        SBT.fit(df[["LIMIT_BAL", "BILL_AMT1"]].values)
