import numpy as np
import pandas as pd
from skorecard.bucket_mapping import BucketMapping


def test_bucket_mapping_numerical():
    """Tests numerical transforms."""
    x = [0, 1, 2, 3, 4, 5]
    bucket = BucketMapping("feature1", "numerical", map=[3, 4])
    assert all(np.equal(bucket.transform(x), np.digitize(x, [3, 4], right=True)))
    # array([0, 0, 0, 0, 1, 2])

    x = [0, 1, 2, 3, 4, 5]
    bucket = BucketMapping("feature1", "numerical", map=[2, 3, 4])
    assert all(np.equal(bucket.transform(x), np.digitize(x, [2, 3, 4], right=True)))
    # array([0, 0, 0, 1, 2, 3])

    x = [0, 1, np.nan, 3, 4, 5]
    bucket = BucketMapping("feature1", "numerical", map=[2, 3, 4])
    assert np.allclose(bucket.transform(x), np.array([0.0, 0.0, np.nan, 1.0, 2.0, 3.0]), equal_nan=True)


def test_bucket_mapping_categorical():
    """Tests categorical transforms."""
    other_category_encoding = -1  # was 'other', but you cannot mix strings and integers.

    # Empty map
    x = ["car", "motorcycle", "boat", "truck", "truck"]
    bucket = BucketMapping("feature1", "categorical", map={})
    assert bucket.transform(x).equals(pd.Series([other_category_encoding] * 5))

    # Empty map with NA's
    x = ["car", "motorcycle", "boat", "truck", "truck", np.nan]
    bucket = BucketMapping("feature1", "categorical", map={})
    assert bucket.transform(x).equals(pd.Series([other_category_encoding] * 5 + [np.nan]))

    # Limited map
    x = ["car", "motorcycle", "boat", "truck", "truck"]
    bucket = BucketMapping("feature1", "categorical", map={"car": 0, "truck": 0})
    reference = pd.Series([0, other_category_encoding, other_category_encoding, 0, 0])
    assert bucket.transform(x).equals(reference)

    # Limited map with NA's
    x = ["car", "motorcycle", "boat", "truck", "truck", np.nan]
    bucket = BucketMapping("feature1", "categorical", map={"car": 0, "truck": 0})
    reference = pd.Series([0, other_category_encoding, other_category_encoding, 0, 0, np.nan])
    assert bucket.transform(x).equals(reference)


def test_get_map():
    """Make sure nicely formatting is returned."""
    bucket = BucketMapping("feature1", "numerical", map=[1, 3, 4], right=True)
    assert bucket.get_map() == ["(-inf, 1.0]", "(1.0, 3.0]", "(3.0, 4.0]", "(4.0, inf]"]

    bucket = BucketMapping("feature1", "numerical", map=[1, 3, 4], right=False)
    assert bucket.get_map() == ["[-inf, 1.0)", "[1.0, 3.0)", "[3.0, 4.0)", "[4.0, inf)"]

    bucket = BucketMapping("feature1", "numerical", map=[1], right=True)
    assert bucket.get_map() == ["(-inf, 1.0]", "(1.0, inf]"]

    bucket = BucketMapping("feature1", "numerical", map=[1], right=False)
    assert bucket.get_map() == ["[-inf, 1.0)", "[1.0, inf)"]

    bucket = BucketMapping("feature1", "numerical", map=[], right=True)
    assert bucket.get_map() == ["(-inf, inf]"]

    bucket = BucketMapping("feature1", "numerical", map=[], right=False)
    assert bucket.get_map() == ["[-inf, inf)"]
