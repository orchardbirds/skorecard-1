"""Classes to store features mapping for bucketing.
"""
import dataclasses
from dataclasses import dataclass, field
from typing import List, Union, Dict, Optional

import pandas as pd
import numpy as np
from probatus.binning import SimpleBucketer

from skorecard.utils import UnknownCategoryError


@dataclass
class BucketMapping:
    """Stores all the info to be able to bucket a feature.

    ```python
    from skorecard.bucket_mapping import BucketMapping

    # Manually a new bucket mapping for a feature
    bucket = BucketMapping('feature1', 'numerical', map = [2,3,4,5])
    print(bucket)

    # You can work with these classes as dicts as well
    bucket.as_dict()
    BucketMapping(**bucket.as_dict())

    # Transform new data
    bucket.transform(list(range(10)))
    ```

    Args:
        feature_name (str): Name of the feature
        type (str): Type of feature, one of ['categorical','numerical']
        map (list or dict): The info needed to create the buckets (boundaries or cats)
        missing_bucket (int): Which bucket to place any missing values
        right (bool): parameter to np.digitize, used when map='numerical'.
    """

    feature_name: str
    type: str
    map: Union[Dict, List] = field(default_factory=lambda: [])
    missing_bucket: Optional[int] = None
    right: bool = True

    def __post_init__(self) -> None:
        """Do input validation.

        Returns:
            None: nothing
        """
        assert self.type in ["numerical", "categorical"]
        # TODO add more assertions here,
        # unique, increasing order, etc
        # missing_bucket index must not be greater than lenght of map

    def transform(self, x):
        """Applies bucketing to and array.

        Args:
            x: array

        Returns:
            x: array
        """
        assert isinstance(x, (list, pd.core.series.Series, np.ndarray))
        assert len(self.map) is not None, "Please set a 'map' first"
        if self.type == "numerical":
            return self._transform_num(x)
        if self.type == "categorical":
            return self._transform_cat(x)

    def _transform_num(self, x):
        """
        Apply binning using a boundaries map.

        Note:
        - We use infinite edges to ensure transformation also works on data outside seen range.
        - np.digitize assigns an extra label for missing values

        ```python
        bins = np.array([-np.inf, 1, np.inf])
        x = np.array([-1,0,.5, 1, 1.5, 2,10, np.nan, 0])
        np.digitize(x, bins)
        ```
        """
        bins = np.hstack(((-np.inf), self.map[1:], (np.inf)))
        buckets = np.digitize(x, bins, right=self.right)
        return buckets.astype(int)

    def _transform_cat(self, x):
        """
        Transforms categorical to buckets.

        Example:
            x: ['a','c','a']
            map: {'a': 0, 'b': 1, 'c': 2}
            output: [0, 2, 0]

        Args:
            x (pd.Series or np.array): Input vector

        """
        assert isinstance(self.map, dict)
        if isinstance(x, np.ndarray):
            x = pd.Series(x)

        if self.missing_bucket is not None:
            mapping = MissingDict(self.map)
            mapping.set_missing_value(self.missing_bucket)
        else:
            mapping = self.map

        x = x.map(mapping)

        if not self.missing_bucket:
            if x.hasnans:
                msg = f"Feature {self.feature_name} has a new, unseen category that causes NaNs."
                msg += "Consider setting 'missing_bucket'."
                raise UnknownCategoryError(msg)

        return x

    def as_dict(self) -> dict:
        """Return data in class as a dict.

        Returns:
            dict: data in class
        """
        return dataclasses.asdict(self)


class FeaturesBucketMapping:
    """Stores a collection of features BucketMapping.

    ```python
    from skorecard.bucket_mapping import FeaturesBucketMapping, BucketMapping

    # Working with collections of BucketMappings
    bucket1 = BucketMapping(feature_name='feature1', type='numerical', map=[2, 3, 4, 5])
    bucket2 = BucketMapping(feature_name='feature2', type='numerical', map=[5,6,7,8])
    features_bucket_mapping = FeaturesBucketMapping([bucket1, bucket2])
    print(features_bucket_mapping)

    # You can also work with class as dict
    features_bucket_mapping.as_dict()

    features_dict = {
        'feature1': {'feature_name': 'feature1',
            'type': 'numerical',
            'map': [2, 3, 4, 5],
            'missing_bucket': None,
            'right': True},
        'feature2': {'feature_name': 'feature2',
            'type': 'numerical',
            'map': [5, 6, 7, 8],
            'missing_bucket': None,
            'right': True}
    }

    features_bucket_mapping = FeaturesBucketMapping()
    features_bucket_mapping.load_dict(features_dict)
    # Or directly from dict
    FeaturesBucketMapping(features_dict)
    ```
    """

    def __init__(self, maps=[]):
        """Takes list of bucketmappings and stores as a dict.

        Args:
            maps (list): list of BucketMapping. Defaults to [].
        """
        self.maps = {}
        if isinstance(maps, list):
            for bucketmap in maps:
                self.append(bucketmap)

        if isinstance(maps, dict):
            for _, bucketmap in maps.items():
                if not isinstance(bucketmap, BucketMapping):
                    bucketmap = BucketMapping(**bucketmap)
                self.append(bucketmap)

    def __repr__(self):
        """Pretty print self.

        Returns:
            str: reproducable object representation.
        """
        class_name = self.__class__.__name__
        maps = list(self.maps.values())
        return f"{class_name}({maps})"

    def get(self, col: str):
        """Get BucketMapping for a column.

        Args:
            col (str): Name of column

        Returns:
            mapping (BucketMapping): BucketMapping for column
        """
        return self.maps[col]

    def append(self, bucketmap: BucketMapping) -> None:
        """Add a BucketMapping to the collection.

        Args:
            bucketmap (BucketMapping): map of a feature
        """
        assert isinstance(bucketmap, BucketMapping)
        self.maps[bucketmap.feature_name] = bucketmap

    def load_yml(self) -> None:
        """Should load in data from a yml.

        Returns:
            None: nothing
        """
        raise NotImplementedError("todo")

    def save_yml(self) -> None:
        """Should write data to a yml.

        Returns:
            None: nothing
        """
        raise NotImplementedError("todo")

    def load_dict(self, obj):
        """Should load in data from a python dict.

        Args:
            obj (dict): Dict with names of features and their BucketMapping

        Returns:
            None: nothing
        """
        assert isinstance(obj, dict)

        self.maps = {}
        for feature, bucketmap in obj.items():
            self.append(BucketMapping(**bucketmap))

    def as_dict(self):
        """Returns data in class as a dict.

        Returns:
            dict: Data in class
        """
        return {k: dataclasses.asdict(v) for k, v in self.maps.items()}


def create_bucket_feature_mapping(df):
    """Simple bucketing on all columns in a DF.

    # TODO: move to binning2d.py. implement wth fit transform

    For some sensible defaults
    """
    features_bucket_mapping = FeaturesBucketMapping()

    for col in df.columns:
        n_bins = min(10, round(df[col].nunique() / 5))
        bucketer = SimpleBucketer(bin_count=n_bins)
        bucketer.fit(df[col])

        features_bucket_mapping.append(BucketMapping(col, "numerical", bucketer.boundaries, right=True))

    return features_bucket_mapping


class MissingDict(dict):
    """Deal with missing values in a dict map.

    Because Pandas .map() uses the __missing__ method
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.map.html

    Example usage:

    ```python
    s = pd.Series(['cat', 'dog', np.nan, 'rabbit'])
    a = {'cat': 'kitten', 'dog': 'puppy'}
    s.map(a)
    a = missingdict(a)
    a.set_missing_value("bye")
    s.map(a)
    ```
    """

    def set_missing_value(self, value):
        """Setter for a missing value."""
        self.missing_value = value

    def __missing__(self, key):
        """Adds a default for missing values."""
        assert self.missing_value is not None, "Use .set_missing_value(key) first"
        return self.missing_value
