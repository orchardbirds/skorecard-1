import numpy as np
import pandas as pd
import pytest
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
    assert np.allclose(bucket.transform(x), np.array([0.0, 0.0, 4, 1.0, 2.0, 3.0]), equal_nan=True)


def test_bucket_mapping_categorical():
    """Tests categorical transforms."""
    # other_category_encoding = -1  # was 'other', but you cannot mix strings and integers.

    # Make sure that the map outputs start at 0 and are incremental. Because it is skipping 2,it will raise an exception
    x = ["car", "motorcycle", "boat", "truck", "truck", np.nan]
    bucket = BucketMapping("feature1", "categorical", map={"car": 0, "truck": 0, "boat": 2})
    with pytest.raises(ValueError):
        bucket.transform(x)

    x = ["car", "motorcycle", "boat", "truck", "truck"]
    bucket = BucketMapping("feature1", "categorical", map={})
    other_category_encoding = 1 if len(bucket.map.values()) == 0 else max(bucket.map.values()) + 1
    assert bucket.transform(x).equals(pd.Series([other_category_encoding] * 5))

    # Empty map with NA's
    x = ["car", "motorcycle", "boat", "truck", "truck", np.nan]
    bucket = BucketMapping("feature1", "categorical", map={})
    other_category_encoding = 1 if len(bucket.map.values()) == 0 else max(bucket.map.values()) + 1
    missing_cat = other_category_encoding + 1
    assert bucket.transform(x).equals(pd.Series([other_category_encoding] * 5 + [missing_cat]))
    #
    # # Limited map
    x = ["car", "motorcycle", "boat", "truck", "truck"]
    bucket = BucketMapping("feature1", "categorical", map={"car": 0, "truck": 0})
    other_category_encoding = 1 if len(bucket.map.values()) == 0 else max(bucket.map.values()) + 1
    reference = pd.Series([0, other_category_encoding, other_category_encoding, 0, 0])
    assert bucket.transform(x).equals(reference)

    # Limited map with NA's
    x = ["car", "motorcycle", "boat", "truck", "truck", np.nan]
    bucket = BucketMapping("feature1", "categorical", map={"car": 0, "truck": 0})
    other_category_encoding = 1 if len(bucket.map.values()) == 0 else max(bucket.map.values()) + 1
    missing_cat = other_category_encoding + 1
    reference = pd.Series([0, other_category_encoding, other_category_encoding, 0, 0, missing_cat])
    assert bucket.transform(x).equals(reference)


def test_specials_numerical():
    """Test that the specials are put in a special bin."""
    # test that a special case
    x = [0, 1, 2, 3, 4, 5, 2]
    bucket = BucketMapping("feature1", "numerical", map=[3, 4], specials={"special": [2]})
    assert all(np.equal(bucket.transform(x), np.array([0, 0, 3, 0, 1, 2, 3])))

    assert bucket.labels == ["(-inf, 3.0]", "(3.0, 4.0]", "(4.0, inf]", "special"]

    # test that calling transform again does not change the labelling
    bucket.transform(x)
    bucket.transform(x)

    assert bucket.labels == ["(-inf, 3.0]", "(3.0, 4.0]", "(4.0, inf]", "special"]

    # Test that if special is not in x, nothing happens
    x = [0, 1, 2, 3, 4, 5]
    bucket = BucketMapping("feature1", "numerical", map=[3, 4], specials={"special": [6]})
    assert all(np.equal(bucket.transform(x), np.digitize(x, [3, 4], right=True)))
