import pytest

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from skorecard import datasets
from skorecard.bucketers import (
    EqualWidthBucketer,
    AgglomerativeClusteringBucketer,
    EqualFrequencyBucketer,
    DecisionTreeBucketer,
    OrdinalCategoricalBucketer,
)

BUCKETERS_WITH_SET_BINS = [
    EqualWidthBucketer,
    AgglomerativeClusteringBucketer,
    EqualFrequencyBucketer,
]

BUCKETERS_AUTO_BINS = [DecisionTreeBucketer, OrdinalCategoricalBucketer]


@pytest.fixture()
def df():
    """Generate dataframe."""
    return datasets.load_uci_credit_card(as_frame=True)


@pytest.mark.parametrize("bucketer", BUCKETERS_WITH_SET_BINS)
def test_fit_x_y(bucketer, df) -> None:
    """Test that using bins=1 puts everything into 1 bucket."""
    X = df
    y = df["default"].values

    BUCK = bucketer(bins=2, variables=["MARRIAGE"])
    BUCK.fit(X, y)
    x_t = BUCK.transform(X)
    assert len(x_t["MARRIAGE"].unique()) == 2


@pytest.mark.parametrize("bucketer", BUCKETERS_WITH_SET_BINS)
def test_single_bucket(bucketer, df) -> None:
    """Test that using bins=1 puts everything into 1 bucket."""
    BUCK = bucketer(bins=1, variables=["MARRIAGE"])
    x_t = BUCK.fit_transform(df)
    assert len(x_t["MARRIAGE"].unique()) == 1


@pytest.mark.parametrize("bucketer", BUCKETERS_WITH_SET_BINS)
def test_three_bins(bucketer, df) -> None:
    """Test that we get the number of bins we request."""
    # Test single bin counts
    BUCK = bucketer(bins=3, variables=["MARRIAGE"])
    x_t = BUCK.fit_transform(df)
    assert len(x_t["MARRIAGE"].unique()) == 3


@pytest.mark.parametrize("bucketer", BUCKETERS_WITH_SET_BINS)
def test_error_input(bucketer, df):
    """Test that a non-int leads to problems in bins."""
    with pytest.raises(AssertionError):
        bucketer(bins=[2])

    with pytest.raises(AssertionError):
        bucketer(bins=4.2, variables=["MARRIAGE"])


@pytest.mark.parametrize("bucketer", BUCKETERS_WITH_SET_BINS)
def test_type_error_input(bucketer, df):
    """Test that input is always a dataFrame."""
    pipe = make_pipeline(StandardScaler(), bucketer(bins=7, variables=["BILL_AMT1"]),)
    with pytest.raises(TypeError):
        pipe.fit_transform(df)
