"""Classes to store features mapping for bucketing.
"""
import dataclasses
from dataclasses import dataclass, field
from typing import List, Union, Dict

import pandas as pd
import numpy as np


@dataclass
class BucketMapping:
    """Stores all the info to be able to bucket a feature.

    ```python
    from skorecard.bucket_mapping import BucketMapping

    # Manually a new bucket mapping for a feature
    bucket = BucketMapping('feature1', 'numerical', map = [2,3,4,5], specials={"special 0": [0]})
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
        right (bool): parameter to np.digitize, used when map='numerical'.
        specials (dict): dictionary of special values to bin separately. The key is used for the bin index,
        labels (dict): dictionary containing special values. It must be of the format:
            - keys: strings, containing the name (that will be used as labels) for the special values
            - values: lists, containing the special values
    """

    feature_name: str
    type: str
    map: Union[Dict, List] = field(default_factory=lambda: [])
    right: bool = True
    specials: Dict = field(default_factory=lambda: {})
    labels: Dict = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        """Do input validation.

        Returns:
            None: nothing
        """
        assert self.type in ["numerical", "categorical", "woe"]
        assert all(
            [isinstance(k, str) for k in self.specials.keys()]
        ), f"The keys of the special dicionary must be \
        strings, got {self.specials.keys()} instead."
        assert all(
            [isinstance(k, list) for k in self.specials.values()]
        ), f"The keys of the special dicionary must be a list of elements, got {self.specials}instead."

    def get_map(self):
        """Pretty form of the boundaries.

        For example [2] will return ["(-inf, 2.0]", "(2.0, inf]"]
        """
        if self.type == "categorical":
            # We might want to implement a nice max_length form here?
            # ['a',...,'z']
            return self.map

        m = [str(c) for c in pd.cut(x=[], bins=[-np.inf] + list(self.map) + [np.inf], right=self.right).categories]

        return m

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
            self._validate_categorical_map()
            return self._transform_cat(x)

        if self.type == "woe":
            return self._transform_cat(x)

    def _validate_categorical_map(self):
        """Assure that the provided mapping starts at 0 and that is tas an incremental trend."""
        values = [v for v in self.map.values()]
        if len(values) > 0:
            if not np.array_equal(np.unique(values), np.arange(max(values) + 1)):
                err_msg = (
                    f"Mapping dictionary must start at 0 and be incremental. "
                    f"Found the following mappings {np.unique(values)}, and expected {np.arange(max(values) + 1)}"
                )
                raise ValueError(err_msg)

    def _transform_num(self, x):
        """
        Apply binning using a boundaries map.

        Note:
        - resulting bins are zero-indexed

        ```python
        import numpy as np
        bins = np.array([-np.inf, 1, np.inf])
        x = np.array([-1,0,.5, 1, 1.5, 2,10, np.nan, 0])
        new = np.digitize(x, bins)
        np.where(np.isnan(x), np.nan, new)
        ```
        """
        self.labels = {}
        if isinstance(x, np.ndarray):
            x = pd.Series(x)
        if isinstance(x, list):
            x = pd.Series(x)

        buckets = self._apply_num_mapping(x)

        max_bucket_number = int(buckets.max())
        # Missings should always have their own bucket number
        # and should come before specials
        self.labels[max_bucket_number + 1] = "Missing"
        if np.isnan(x).any():
            buckets = np.where(np.isnan(x), max_bucket_number + 1, buckets)

        if len(self.specials) > 0:
            buckets = self._assign_specials(x, buckets, start_bucket_number=max_bucket_number + 1)

        return buckets

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
        if isinstance(x, list):
            x = pd.Series(x)

        new = self._apply_cat_mapping(x)

        max_bucket_number = int(max(self.labels.keys()))
        # Missings should always have their own bucket number
        # and should come before specials
        self.labels[max_bucket_number + 1] = "Missing"
        if x.isna().any():
            new = pd.Series(np.where(x.isna(), max_bucket_number + 1, new))

        if len(self.specials) > 0:
            new = self._assign_specials(x, new, start_bucket_number=max_bucket_number + 1)

        return new

    def _apply_cat_mapping(self, x):
        other_value = 1 if len(self.map.values()) == 0 else max(self.map.values()) + 1
        mapping = MissingDict(self.map)
        mapping.set_missing_value(other_value)  # This was 'other' but you cannot mix integers and strings

        new = x.map(mapping)

        # define the labels
        v = {}

        # create a dictionary that groups by the
        if 0 not in mapping.values():
            v[0] = "empty map"
        else:
            for key, value in sorted(mapping.items()):
                if not isinstance(key, str):
                    key = str(key)
                v.setdefault(value, []).append(key)
        v[other_value] = "other"
        sorted_v = {key: v[key] for key in sorted(v)}

        # transfrom it all to a string like format
        for k, v in sorted_v.items():

            # if k is of type int32, it will create weird caracters if exported to files
            # if self.type=='categorical':
            if isinstance(k, np.int32):
                k = int(k)
            if isinstance(v, list):
                if len(v) == 1:
                    v = v[0]
                    if not isinstance(v, str):
                        v = str(v)
                else:
                    v = ", ".join(v)
            sorted_v[k] = v

        self.labels = sorted_v

        return new

    def as_dict(self) -> dict:
        """Return data in class as a dict.

        Returns:
            dict: data in class
        """
        return dataclasses.asdict(self)

    def _apply_num_mapping(self, x):
        """Apply numerical bucketing and stores the labels for the buckets.

        Args:
            x (np.array): feature

        Returns:
            (np.array), buckets
        """
        buckets = np.digitize(x, self.map, right=self.right)
        buckets = buckets.astype(int)

        map_ = np.hstack([-np.inf, self.map, np.inf])

        for bucket in np.unique(buckets):
            bucket_str = f"{map_[int(bucket)]}, {map_[int(bucket) + 1]}"
            if self.right:
                # The infinite edge should not be inclusive
                if not bucket_str.endswith("inf"):
                    bucket_str = f"({bucket_str}]"
                else:
                    bucket_str = f"({bucket_str})"

            else:
                if not bucket_str.startswith("-inf"):
                    bucket_str = f"[{bucket_str})"
                else:
                    bucket_str = f"({bucket_str})"

            self.labels[bucket] = bucket_str

        return buckets

    def _assign_specials(self, x, buckets, start_bucket_number=None):
        """Assign the special buckets as defined in the specials dictionary.

        Args:
            x (np.array): feature
            buckets (np.array): the bucketed x
            start_bucket_number (int): where to start numbering the

        Returns:
            (np.array), buckets
        """
        if not start_bucket_number:
            start_bucket_number = int(buckets.max())

        for k, v in self.specials.items():
            start_bucket_number += 1
            buckets = np.where(x.isin(v), start_bucket_number, buckets)
            self.labels[start_bucket_number] = "Special: " + str(k)
        return buckets


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
            'right': True},
        'feature2': {'feature_name': 'feature2',
            'type': 'numerical',
            'map': [5, 6, 7, 8],
            'right': True}
    }

    features_bucket_mapping = FeaturesBucketMapping()
    features_bucket_mapping.load_dict(features_dict)
    # Or directly from dict
    FeaturesBucketMapping(features_dict)
    # See columns
    features_bucket_mapping.columns
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

    @property
    def columns(self):
        """Returns the columns that have a bucket_mapping."""
        return list(self.as_dict().keys())


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
