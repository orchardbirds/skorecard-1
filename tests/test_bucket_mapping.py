import numpy as np
import pandas as pd
from skorecard.bucket_mapping import BucketMapping


def test_bucket_mapping_numerical():
    """Tests numerical transforms."""
    x = [0, 1, 2, 3, 4, 5]
    bucket = BucketMapping("feature1", "numerical", map=[3, 4])
    assert all(np.equal(bucket.transform(x), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])))
    # np.digitize(x, [3,4], right=True)
    # array([0, 0, 0, 0, 1, 2])

    x = [0, 1, 2, 3, 4, 5]
    bucket = BucketMapping("feature1", "numerical", map=[2, 3, 4])
    assert all(np.equal(bucket.transform(x), np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0])))
    # np.digitize(x, [2, 3, 4], right=True)
    # array([0, 0, 0, 1, 2, 3])

    x = [0, 1, np.nan, 3, 4, 5]
    bucket = BucketMapping("feature1", "numerical", map=[2, 3, 4])
    assert np.allclose(bucket.transform(x), np.array([1.0, 1.0, np.nan, 1.0, 2.0, 2.0]), equal_nan=True)
    # np.digitize(x, [2, 3, 4], right=True)
    # array([0, 0, 3, 1, 2, 3])


def test_bucket_mapping_categorical():
    """Tests categorical transforms."""
    # Empty map
    x = ["car", "motorcycle", "boat", "truck", "truck"]
    bucket = BucketMapping("feature1", "categorical", map={})
    assert bucket.transform(x).equals(pd.Series(["other"] * 5))

    # Empty map with NA's
    x = ["car", "motorcycle", "boat", "truck", "truck", np.nan]
    bucket = BucketMapping("feature1", "categorical", map={})
    assert bucket.transform(x).equals(pd.Series(["other"] * 5 + [np.nan]))

    # Limited map
    x = ["car", "motorcycle", "boat", "truck", "truck"]
    bucket = BucketMapping("feature1", "categorical", map={"car": 0, "truck": 0})
    reference = pd.Series([0, "other", "other", 0, 0])
    assert bucket.transform(x).equals(reference)

    # Limited map with NA's
    x = ["car", "motorcycle", "boat", "truck", "truck", np.nan]
    bucket = BucketMapping("feature1", "categorical", map={"car": 0, "truck": 0})
    reference = pd.Series([0, "other", "other", 0, 0, np.nan])
    assert bucket.transform(x).equals(reference)
