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
    assert "Missing" in bucket.labels.values()


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
    assert all(np.equal(bucket.transform(x), [other_category_encoding] * 5))

    # Empty map with NA's
    x = ["car", "motorcycle", "boat", "truck", "truck", np.nan]
    bucket = BucketMapping("feature1", "categorical", map={})
    other_category_encoding = 1 if len(bucket.map.values()) == 0 else max(bucket.map.values()) + 1
    missing_cat = other_category_encoding + 1
    assert all(np.equal(bucket.transform(x), [other_category_encoding] * 5 + [missing_cat]))
    #
    # # Limited map
    x = ["car", "motorcycle", "boat", "truck", "truck"]
    bucket = BucketMapping("feature1", "categorical", map={"car": 0, "truck": 0})
    other_category_encoding = 1 if len(bucket.map.values()) == 0 else max(bucket.map.values()) + 1
    reference = [0, other_category_encoding, other_category_encoding, 0, 0]
    assert all(np.equal(bucket.transform(x), reference))

    # Limited map with NA's
    x = ["car", "motorcycle", "boat", "truck", "truck", np.nan]
    bucket = BucketMapping("feature1", "categorical", map={"car": 0, "truck": 0})
    other_category_encoding = 1 if len(bucket.map.values()) == 0 else max(bucket.map.values()) + 1
    missing_cat = other_category_encoding + 1
    reference = [0, other_category_encoding, other_category_encoding, 0, 0, missing_cat]
    assert all(np.equal(bucket.transform(x), reference))


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


def test_specials_numerical():
    """Test that the specials are put in a special bin."""
    # test that a special case
    x = [0, 1, 2, 3, 4, 5, 2]
    bucket = BucketMapping("feature1", "numerical", map=[3, 4], specials={"=2": [2]})
    assert all(np.equal(bucket.transform(x), np.array([0, 0, 4, 0, 1, 2, 4])))

    assert bucket.labels == {0: "(-inf, 3.0]", 1: "(3.0, 4.0]", 2: "(4.0, inf)", 3: "Missing", 4: "Special: =2"}

    # test that calling transform again does not change the labelling
    bucket.transform(x)
    bucket.transform(x)

    assert bucket.labels == {0: "(-inf, 3.0]", 1: "(3.0, 4.0]", 2: "(4.0, inf)", 3: "Missing", 4: "Special: =2"}

    # Test that if special is not in x, nothing happens
    x = [0, 1, 2, 3, 4, 5]
    bucket = BucketMapping("feature1", "numerical", map=[3, 4], specials={"=6": [6]})
    assert all(np.equal(bucket.transform(x), np.digitize(x, [3, 4], right=True)))


def test_labels():
    """Test that the labels are correct in different scenarios."""
    x = ["car", "motorcycle", "boat", "truck", "truck", np.nan]
    bucket = BucketMapping("feature1", "categorical", map={"car": 0, "boat": 0}, specials={"is truck": ["truck"]})
    bins = pd.Series(bucket.transform(x))

    in_series = pd.Series(x)

    labels = bins.map(bucket.labels)

    assert labels[in_series == "truck"].equals(labels[labels == "Special: is truck"])
    assert labels[(in_series.isin(["car", "boat"]))].equals(labels[labels == "boat, car"])

    # test with numerical categories
    # Limited map with NA's
    x = [310, 311, 312, 313, 313, np.nan]
    bucket = BucketMapping("feature1", "categorical", map={310: 0, 311: 1, 312: 2}, specials={"is 313": [313]})
    bins = pd.Series(bucket.transform(x))

    in_series = pd.Series(x)

    labels = bins.map(bucket.labels)

    assert labels[in_series == 313].equals(labels[labels == "Special: is 313"])
    assert labels[in_series == 311].equals(labels[labels == "311"])
    assert labels[in_series.isna()].equals(labels[labels == "Missing"])

    # test numerical labels
    x = [0, 1, 2, 3, 4, 5, 2, np.nan]
    bucket = BucketMapping("feature1", "numerical", map=[3, 4], specials={"=2": [2]})
    bins = pd.Series(bucket.transform(x))
    in_series = pd.Series(x)

    labels = bins.map(bucket.labels)

    assert labels[(in_series <= 3) & (in_series != 2)].equals(labels[labels == "(-inf, 3.0]"])
    assert labels[(in_series <= 4) & (in_series > 3)].equals(labels[labels == "(3.0, 4.0]"])
    assert labels[in_series == 2].equals(labels[labels == "Special: =2"])
    assert labels[in_series > 4].equals(labels[labels == "(4.0, inf)"])


def test_error_is_raised_if_wrong_specials():
    """Self explanatory."""
    # Test that is the values of the dictionary
    with pytest.raises(AssertionError):
        BucketMapping("feature1", "numerical", map=[3, 4], specials={"special": 2})
    #
    with pytest.raises(AssertionError):
        BucketMapping("feature1", "numerical", map=[3, 4], specials={0: [2]})
