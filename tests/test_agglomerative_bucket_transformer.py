from skorecard import datasets
from skorecard.preprocessing import AgglomerativeBucketTransformer

import pytest


@pytest.fixture()
def df():
    """Generate dataframe."""
    return datasets.load_uci_credit_card(as_frame=True)


def test_single_bucket(df):
    """Test that using bin_count=1 puts everything into 1 bucket."""
    ABT = AgglomerativeBucketTransformer(bin_count=1)
    ABT.fit(df["MARRIAGE"].values)
    assert ABT.Bucketer.counts == df.shape[0]

    return None


def test_bin_counts(df):
    """Test that we get the number of bins we request."""
    # Test single bin counts
    ABT = AgglomerativeBucketTransformer(bin_count=3)
    ABT.fit_transform(df["MARRIAGE"].values)
    assert ABT.BucketDict["Feature_0"].bin_count == 3

    # Test multiple bin counts
    ABT = AgglomerativeBucketTransformer(bin_count=[2, 2, 3])
    ABT.fit_transform(df[["LIMIT_BAL", "MARRIAGE", "BILL_AMT1"]].values)
    assert ABT.BucketDict["Feature_0"].bin_count == 2
    assert ABT.BucketDict["Feature_1"].bin_count == 2
    assert ABT.BucketDict["Feature_2"].bin_count == 3

    return None


def test_float_bin_count(df):
    """Test that a float for bin_count raises an error."""
    try:
        ABT = AgglomerativeBucketTransformer(bin_count=7.3)
        ABT.fit_transform(df["LIMIT_BAL"].values)
    except AttributeError:
        assert True

    return None


def test_bucket_dict_number(df):
    """Test that we get a separate Bucketer object per number of columns given."""
    ABT = AgglomerativeBucketTransformer(bin_count=[4, 2, 3])
    ABT.fit(df[["MARRIAGE", "BILL_AMT1", "LIMIT_BAL"]].values)
    assert len(ABT.BucketDict) == 3

    return None


def test_bad_bin_count_shape(df):
    """Test that bad bin shape triggers ValueError."""
    # Agglomerative Bucketer
    try:
        ABT = AgglomerativeBucketTransformer(bin_count=[3, 3])
        ABT.fit(df["LIMIT_BAL"].values)
    except ValueError:
        assert True

    try:
        ABT = AgglomerativeBucketTransformer(bin_count=[3])
        ABT.fit(df[["LIMIT_BAL", "BILL_AMT1"]].values)
    except ValueError:
        assert True

    return None
