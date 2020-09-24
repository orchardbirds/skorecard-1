from skorecard import datasets
from skorecard.preprocessing import SimpleBucketTransformer, AgglomerativeBucketTransformer, QuantileBucketTransformer
from skorecard.utils.exceptions import DimensionalityError

import numpy as np
import pytest


TRANSFORMERS = [SimpleBucketTransformer, AgglomerativeBucketTransformer, QuantileBucketTransformer]


@pytest.fixture()
def df():
    """Generate dataframe."""
    return datasets.load_uci_credit_card(as_frame=True)


@pytest.mark.parametrize("bucketer", TRANSFORMERS)
def test_single_bucket(bucketer, df) -> None:
    """Test that using bin_count=1 puts everything into 1 bucket."""
    BUCK = bucketer(bin_count=1)
    x_t = BUCK.fit_transform(df["MARRIAGE"].values)
    assert len(np.unique(x_t)) == 1


@pytest.mark.parametrize("bucketer", TRANSFORMERS)
def test_bin_counts(bucketer, df) -> None:
    """Test that we get the number of bins we request."""
    # Test single bin counts
    BUCK = AgglomerativeBucketTransformer(bin_count=3)
    x_t = BUCK.fit_transform(df["MARRIAGE"].values)
    assert len(np.unique(x_t)) == 3


@pytest.mark.parametrize("bucketer", TRANSFORMERS)
def test_list_bin_counts(bucketer, df):
    """Test that a non-int leads to problems in bin_count."""
    with pytest.raises(AttributeError):
        bucketer(bin_count=[2])


@pytest.mark.parametrize("bucketer", TRANSFORMERS)
def test_dimensionality_exception(bucketer, df):
    """Test the exception is raised if too many features are run at the same time."""
    BUCK = AgglomerativeBucketTransformer(bin_count=2)
    with pytest.raises(DimensionalityError):
        BUCK.fit_transform(df[["LIMIT_BAL", "MARRIAGE"]].values)

    return None


@pytest.mark.parametrize("bucketer", TRANSFORMERS)
def test_float_bin_count(bucketer, df):
    """Test that a float for bin_count raises an error."""
    with pytest.raises(AttributeError):
        BUCK = AgglomerativeBucketTransformer(bin_count=7.3)
        BUCK.fit_transform(df["LIMIT_BAL"].values)

    return None
