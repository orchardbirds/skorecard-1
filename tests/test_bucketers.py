import pytest
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from skorecard.bucketers import (
    EqualWidthBucketer,
    AgglomerativeClusteringBucketer,
    EqualFrequencyBucketer,
)

BUCKETERS_WITH_SET_BINS = [
    EqualWidthBucketer,
    AgglomerativeClusteringBucketer,
    EqualFrequencyBucketer,
]


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
    df = df.drop(columns=["pet_ownership"])
    pipe = make_pipeline(StandardScaler(), bucketer(bins=7, variables=["BILL_AMT1"]),)
    with pytest.raises(TypeError):
        pipe.fit_transform(df)


@pytest.mark.parametrize("bucketer", BUCKETERS_WITH_SET_BINS)
def test_zero_indexed(bucketer, df):
    """Test that bins are zero-indexed."""
    BUCK = bucketer(bins=3)
    x_t = BUCK.fit_transform(df.drop(columns=["pet_ownership"]))
    assert x_t["MARRIAGE"].min() == 0
    assert x_t["EDUCATION"].min() == 0
    assert x_t["LIMIT_BAL"].min() == 0
    assert x_t["BILL_AMT1"].min() == 0


@pytest.mark.parametrize("bucketer", BUCKETERS_WITH_SET_BINS)
def test_missing_default(bucketer, df_with_missings) -> None:
    """Test that missing values are assigned to the right bucket."""
    X = df_with_missings
    y = df_with_missings["default"].values

    BUCK = bucketer(bins=2, variables=["MARRIAGE"])
    BUCK.fit(X, y)
    X['MARRIAGE_trans'] = BUCK.transform(X[['MARRIAGE']])
    assert len(X["MARRIAGE_trans"].unique()) == 3
    assert X[np.isnan(X['MARRIAGE'])].shape[0] == X[X['MARRIAGE_trans'] == 2].shape[0]