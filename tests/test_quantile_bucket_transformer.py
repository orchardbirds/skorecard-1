from skorecard import datasets
from skorecard.preprocessing import QuantileBucketTransformer
from skorecard.utils.exceptions import DimensionalityError

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
    QBT = QuantileBucketTransformer(bin_count=3)
    QBT.fit_transform(df["MARRIAGE"].values)
    assert QBT.BucketDict["QuantileBucketer"].bin_count == 3

    return None


def test_list_bin_counts(df):
    """Test that a non-int leads to problems in bin_count."""
    with pytest.raises(AttributeError):
        QuantileBucketTransformer(bin_count=[2])

    return None


def test_float_bin_count(df):
    """Test that a float for bin_count raises an error."""
    with pytest.raises(AttributeError):
        QBT = QuantileBucketTransformer(bin_count=7.3)
        QBT.fit_transform(df["LIMIT_BAL"].values)

    return None


def test_dimensionality_exception(df):
    """Test the exception is raised if too many features are run at the same time."""
    QBT = QuantileBucketTransformer(bin_count=2)
    with pytest.raises(DimensionalityError):
        QBT.fit_transform(df[["LIMIT_BAL", "MARRIAGE"]].values)

    return None
