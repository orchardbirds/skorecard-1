from skorecard import datasets
from skorecard.preprocessing import AgglomerativeBucketTransformer
from skorecard.utils.exceptions import DimensionalityError

import numpy as np
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
    assert ABT.BucketDict["AgglomerativeBucketer"].bin_count == 3

    return None


def test_list_bin_counts(df):
    """Test that a non-int leads to problems in bin_count."""
    with pytest.raises(AttributeError):
        AgglomerativeBucketTransformer(bin_count=[2])

    return None


def test_dimensionality_exception(df):
    """Test the exception is raised if too many features are run at the same time."""
    ABT = AgglomerativeBucketTransformer(bin_count=2)
    with pytest.raises(DimensionalityError):
        ABT.fit_transform(df[["LIMIT_BAL", "MARRIAGE"]].values)

    return None


def test_float_bin_count(df):
    """Test that a float for bin_count raises an error."""
    with pytest.raises(AttributeError):
        ABT = AgglomerativeBucketTransformer(bin_count=7.3)
        ABT.fit_transform(df["LIMIT_BAL"].values)

    return None


def test_inifinte_edges(df):
    """Test that infinite edges actually are infinite."""
    SBT = AgglomerativeBucketTransformer(bin_count=5, infinite_edges=True)
    SBT.fit(df["LIMIT_BAL"].values)
    assert SBT.boundaries[0] == -np.inf
    assert SBT.boundaries[-1] == np.inf

    SBT = AgglomerativeBucketTransformer(bin_count=5, infinite_edges=False)
    SBT.fit(df["LIMIT_BAL"].values)
    assert SBT.boundaries[0] != -np.inf
    assert SBT.boundaries[-1] != np.inf
