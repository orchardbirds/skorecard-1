import numpy as np
from skorecard import datasets
from skorecard.preprocessing import SimpleBucketTransformer
from skorecard.preprocessing import AgglomerativeBucketTransformer
from skorecard.preprocessing import QuantileBucketTransformer
from skorecard.preprocessing import ManualBucketTransformer


import pytest


@pytest.fixture()
def df():
    """Generate dataframe."""
    return datasets.load_uci_credit_card(as_frame=True)


@pytest.fixture()
def example_boundary_dict():
    """Generate example dict."""
    return {
        1: [0.0, 1.5, 3.0],
        0: [0.0, 1.5, 2.5, 4.5, 6.0],
        2: [10000.0, 30000.0, 50000.0, 70000.0, 100000.0, 140000.0, 180000.0, 220000.0, 280000.0, 360000.0, 760000.0],
        3: [-165580.0, 50000, 610723.0],
    }


def test_single_bucket(df):
    """Test that using bin_count=1 puts everything into 1 bucket."""
    SBT = SimpleBucketTransformer(bin_count=1)
    SBT.fit(df["LIMIT_BAL"].values)
    assert SBT.Bucketer.counts == df.shape[0]
    ABT = AgglomerativeBucketTransformer(bin_count=1)
    ABT.fit(df["MARRIAGE"].values)
    assert ABT.Bucketer.counts == df.shape[0]
    QBT = QuantileBucketTransformer(bin_count=1)
    QBT.fit(df["EDUCATION"].values)
    assert QBT.Bucketer.counts == df.shape[0]

    return None


def test_bin_counts(df):
    """Test that we get the number of bins we request."""
    # Test single bin counts
    SBT = SimpleBucketTransformer(bin_count=7)
    X = SBT.fit_transform(df["LIMIT_BAL"].values)
    assert len(np.unique(X)) == 7
    ABT = AgglomerativeBucketTransformer(bin_count=3)
    X = ABT.fit_transform(df["MARRIAGE"].values)
    assert len(np.unique(X)) == 3
    QBT = QuantileBucketTransformer(bin_count=10)
    X = QBT.fit_transform(df["BILL_AMT1"].values)
    assert len(np.unique(X)) == 10

    # Test multiple bin counts
    SBT = SimpleBucketTransformer(bin_count=[7, 5])
    X = SBT.fit_transform(df[["LIMIT_BAL", "BILL_AMT1"]].values)
    assert len(np.unique(X[:, 0])) == 7
    assert len(np.unique(X[:, 1])) == 5
    ABT = AgglomerativeBucketTransformer(bin_count=[2, 2, 3])
    X = ABT.fit_transform(df[["LIMIT_BAL", "MARRIAGE", "BILL_AMT1"]].values)
    assert len(np.unique(X[:, 0])) == 2
    assert len(np.unique(X[:, 1])) == 2
    assert len(np.unique(X[:, 2])) == 3
    QBT = QuantileBucketTransformer(bin_count=[4, 3, 2, 6])
    X = QBT.fit_transform(df[["LIMIT_BAL", "MARRIAGE", "EDUCATION", "BILL_AMT1"]].values)
    assert len(np.unique(X[:, 0])) == 4
    assert len(np.unique(X[:, 1])) == 3
    assert len(np.unique(X[:, 2])) == 2
    assert len(np.unique(X[:, 3])) == 6

    return None


def test_float_bin_count(df):
    """Test that a float for bin_count raises an error."""
    try:
        SBT = SimpleBucketTransformer(bin_count=7.3)
        SBT.fit_transform(df["LIMIT_BAL"].values)
    except AttributeError:
        assert True

    try:
        ABT = AgglomerativeBucketTransformer(bin_count=7.3)
        ABT.fit_transform(df["LIMIT_BAL"].values)
    except AttributeError:
        assert True

    try:
        QBT = QuantileBucketTransformer(bin_count=7.3)
        QBT.fit_transform(df["LIMIT_BAL"].values)
    except AttributeError:
        assert True

    return None


def test_bucket_dict_number(df):
    """Test that we get a separate Bucketer object per number of columns given."""
    SBT = SimpleBucketTransformer(bin_count=[3, 3])
    SBT.fit(df[["LIMIT_BAL", "BILL_AMT1"]].values)
    assert len(SBT.BucketDict) == 2
    ABT = AgglomerativeBucketTransformer(bin_count=[4, 2, 3])
    ABT.fit(df[["MARRIAGE", "BILL_AMT1", "LIMIT_BAL"]].values)
    assert len(ABT.BucketDict) == 3
    QBT = QuantileBucketTransformer(bin_count=[2, 3, 2, 3])
    QBT.fit(df[["LIMIT_BAL", "BILL_AMT1", "LIMIT_BAL", "EDUCATION"]].values)
    assert len(QBT.BucketDict) == 4

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

    # Agglomerative Bucketer
    try:
        ABT = AgglomerativeBucketTransformer(bin_count=[3, 3])
        ABT.fit(df["LIMIT_BAL"].values)
    except ValueError:
        assert True

    try:
        SBT = SimpleBucketTransformer(bin_count=[3])
        SBT.fit(df[["LIMIT_BAL", "BILL_AMT1"]].values)
    except ValueError:
        assert True

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


def test_example_boundary_dict(df, example_boundary_dict):
    """Test that we can use an example dict for ManualBucketTransformer."""
    MBT = ManualBucketTransformer(boundary_dict=example_boundary_dict)
    X = MBT.fit_transform(df.values)

    # We do not test features 0 and 1 yet as they are categoricals
    assert len(np.unique(X[:, 2])) == len(example_boundary_dict[2]) - 1
    assert len(np.unique(X[:, 3])) == len(example_boundary_dict[3]) - 1
