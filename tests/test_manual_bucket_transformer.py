import numpy as np
from skorecard import datasets
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


@pytest.fixture()
def one_feat_dictionary():
    """Returns dictionary for one feature only."""
    return {
        1: [0.0, 1.5, 3.0],
        2: [10000.0, 30000.0, 140000.0, 360000.0, 760000.0],
    }


def test_example_boundary_dict(df, example_boundary_dict):
    """Test that we can use an example dict for ManualBucketTransformer."""
    MBT = ManualBucketTransformer(boundary_dict=example_boundary_dict)
    X = MBT.fit_transform(df.values)

    # We do not test features 0 and 1 yet as they are categoricals
    assert len(np.unique(X[:, 2])) == len(example_boundary_dict[2])
    assert len(np.unique(X[:, 3])) == len(example_boundary_dict[3])


def test_only_one_feat_in_dict(df, one_feat_dictionary):
    """Test that when not all the features are defined, only those defined get bucketed."""
    MBT = ManualBucketTransformer(boundary_dict=one_feat_dictionary)
    X = MBT.fit_transform(df.values)

    # We do not test features 0 and 1 yet as they are categoricals
    assert len(np.unique(X[:, 1])) == len(one_feat_dictionary[1])
    assert len(np.unique(X[:, 2])) == len(one_feat_dictionary[2])
