"""Classes to store features mapping for bucketing.

Examples:
    # Create a new bucket mapping for a feature
    bucket = BucketMapping('feature1', 'numerical', map = [2,3,4,5])
    bucket

    # Working with collections of BucketMappings
    features_bucket_mapping = FeaturesBucketMapping([bucket])
    features_bucket_mapping
    features_bucket_mapping.as_dict()
"""
from dataclasses import dataclass, field
import dataclasses
from typing import List, Optional
import pandas as pd
import numpy as np

from probatus.binning import SimpleBucketer


class FeaturesBucketMapping:
    """Stores a collection of features BucketMapping.

    That's it.
    """

    def __init__(self, maps=[]):
        """Takes list of bucketmappings and stores a dict.

        Args:
            maps (list): list of BucketMapping. Defaults to [].
        """
        self.maps = {}

        if len(maps) > 0:
            for bucketmap in maps:
                self.append(bucketmap)

    def __repr__(self):
        """Pretty print self.

        Returns:
            str: reproducable object representation.
        """
        return "%s(%s)" % (self.__class__.__name__, self.maps)

    def get(self, col: str):
        """Get BucketMapping for a column.

        Args:
            col (str): Name of column

        Returns:
            mapping (BucketMapping): BucketMapping for column
        """
        return self.maps[col]

    def append(self, bucketmap) -> None:
        """Add a BucketMapping to the collection.

        Args:
            bucketmap (BucketMapping): map of a feature
        """
        if not isinstance(bucketmap, BucketMapping):
            bucketmap = BucketMapping(**bucketmap)
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
        raise NotImplementedError("todo")

    def as_dict(self):
        """Returns data in class as a dict.

        Returns:
            dict: Data in class
        """
        return {k: dataclasses.asdict(v) for k, v in self.maps.items()}


@dataclass
class BucketMapping:
    """Stores all the info to be able to bucket a feature.

    Args:
        feature_name (str): Name of the feature
        # TODO
    """

    feature_name: str
    type: str
    map: List = field(default_factory=lambda: [])
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
        return np.digitize(x, self.map[1:], right=self.right)

    def _transform_cat(self, x):
        """
        Transforms categorical to buckets.

        Example:
            input ['a','c']
            map [['a','b'],['c','d']]
            output [0, 1]

        Args:
            x ([type]): [description]

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError("We need to find a numpy or pandas function for this one.")

    def as_dict(self) -> dict:
        """Return data in class as a dict.

        Returns:
            dict: data in class
        """
        return dataclasses.asdict(self)


def create_bucket_feature_mapping(df):
    """Simple bucketing on all columns in a DF.

    For some sensible defaults
    """
    features_bucket_mapping = FeaturesBucketMapping()

    for col in df.columns:
        n_bins = min(10, round(df[col].nunique() / 5))
        bucketer = SimpleBucketer(bin_count=n_bins)
        bucketer.fit(df[col])

        features_bucket_mapping.append(BucketMapping(col, "numerical", bucketer.boundaries, right=True))

    return features_bucket_mapping
