import numpy as np
from sklearn.compose import ColumnTransformer
from skorecard import datasets
from skorecard.preprocessing import SimpleBucketTransformer
from skorecard.preprocessing import AgglomerativeBucketTransformer
from skorecard.preprocessing import QuantileBucketTransformer
import pytest
from skorecard.utils.exceptions import DimensionalityError


@pytest.fixture()
def df():
    """Generate dataframe."""
    return datasets.load_uci_credit_card(as_frame=True)


def test_bucket_transformer_exception(df):
    """Test the exception is raised in scikit-learn pipeline."""
    transformer = ColumnTransformer(
        transformers=[
            ("simple", SimpleBucketTransformer(bin_count=2), [1]),
            ("agglom", AgglomerativeBucketTransformer(bin_count=4), [0]),
            ("quantile", QuantileBucketTransformer(bin_count=[10, 6]), [2, 3]),
        ],
        remainder="passthrough",
    )
    with pytest.raises(DimensionalityError):
        transformer.fit_transform(df.values)


def test_bucket_transformer(df):
    """Test that we can utilise the main bucket transformers in a scikit-learn pipeline."""
    transformer = ColumnTransformer(
        transformers=[
            ("simple", SimpleBucketTransformer(bin_count=2), [1]),
            ("agglom", AgglomerativeBucketTransformer(bin_count=4), [0]),
            ("quantile_0", QuantileBucketTransformer(bin_count=[10]), [2]),
            ("quantile_1", QuantileBucketTransformer(bin_count=[6]), [3]),
        ],
        remainder="passthrough",
    )

    X = transformer.fit_transform(df.values)

    # Test only non-categorical variables
    assert len(np.unique(X[:, 2])) == 11
    assert len(np.unique(X[:, 3])) == 7
    assert np.all(X[:, 4] == df["default"].values)

    return None
