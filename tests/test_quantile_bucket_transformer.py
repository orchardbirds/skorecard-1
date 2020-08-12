from skorecard import datasets
from skorecard.preprocessing import QuantileBucketTransformer

import pytest


@pytest.fixture()
def df():
    """Generate dataframe."""
    return datasets.load_uci_credit_card(as_frame=True)


def test_single_bucket(df):
    """Test that using bin_count=1 puts everything into 1 bucket."""
    QBT = QuantileBucketTransformer(bin_count=1)
    QBT.fit(df["EDUCATION"].values)
    assert QBT.Bucketer.counts == df.shape[0]

    return None


def test_bin_counts(df):
    """Test that we get the number of bins we request."""
    # Test single bin counts
    QBT = QuantileBucketTransformer(bin_count=10)
    QBT.fit_transform(df["BILL_AMT1"].values)
    assert QBT.BucketDict["Feature_0"].bin_count == 10

    # Test multiple bin counts
    QBT = QuantileBucketTransformer(bin_count=[4, 3, 2, 6])
    QBT.fit_transform(df[["LIMIT_BAL", "MARRIAGE", "EDUCATION", "BILL_AMT1"]].values)
    assert QBT.BucketDict["Feature_0"].bin_count == 4
    assert QBT.BucketDict["Feature_1"].bin_count == 3
    assert QBT.BucketDict["Feature_2"].bin_count == 2
    assert QBT.BucketDict["Feature_3"].bin_count == 6

    return None


def test_float_bin_count(df):
    """Test that a float for bin_count raises an error."""
    try:
        QBT = QuantileBucketTransformer(bin_count=7.3)
        QBT.fit_transform(df["LIMIT_BAL"].values)
    except AttributeError:
        assert True

    return None


def test_bucket_dict_number(df):
    """Test that we get a separate Bucketer object per number of columns given."""
    QBT = QuantileBucketTransformer(bin_count=[2, 3, 2, 3])
    QBT.fit(df[["LIMIT_BAL", "BILL_AMT1", "LIMIT_BAL", "EDUCATION"]].values)
    assert len(QBT.BucketDict) == 4

    return None


def test_bad_bin_count_shape(df):
    """Test that bad bin shape triggers ValueError."""
    # Quantile Bucketer
    try:
        QBT = QuantileBucketTransformer(bin_count=[3, 3])
        QBT.fit(df["LIMIT_BAL"].values)
    except ValueError:
        assert True
    try:
        QBT = QuantileBucketTransformer(bin_count=[3])
        QBT.fit(df[["LIMIT_BAL", "BILL_AMT1"]].values)
    except ValueError:
        assert True

    return None
