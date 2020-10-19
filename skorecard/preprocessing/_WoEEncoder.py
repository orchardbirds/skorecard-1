import numpy as np

from skorecard.bucketers.base_bucketer import BaseBucketer
from skorecard.bucket_mapping import BucketMapping
from skorecard.metrics.metrics import woe_1d


class WoeEncoder(BaseBucketer):
    """Transformer that encodes unique values in features to their Weight of Evidence estimation.

    Only works for binary classification.

    The weight of evidence is given by: `np.log( p(1) / p(0) )`
    The target probability ratio is given by: `p(1) / p(0)`

    For example in the variable colour, if the mean of the target = 1 for blue is 0.8 and
    the mean of the target = 0 is 0.2, blue will be replaced by: np.log(0.8/0.2) = 1.386
    if log_ratio is selected. Alternatively, blue will be replaced by 0.8 / 0.2 = 4 if ratio is selected.

    For details on the weight of evidence:
    https://multithreaded.stitchfix.com/blog/2015/08/13/weight-of-evidence/

    ```python
    from skorecard import datasets
    from skorecard.preprocessing import WoeEncoder

    X, y = datasets.load_uci_credit_card(return_X_y=True)
    we = WoeEncoder(variables=['EDUCATION'])
    we.fit_transform(X, y)
    we.fit_transform(X, y)['EDUCATION'].value_counts()
    ```

    Credit: Some code taken from feature_engine.categorical_encoders.

    """

    def __init__(self, epsilon=0.0001, variables=[]):
        """Constructor for WoEBucketer.

        Args:
            epsilon (float): Amount to be added to relative counts in order to avoid division by zero in the WOE
            calculation.
            variables (list): The features to bucket. Uses all features if not defined.
        """
        assert isinstance(variables, list)
        assert epsilon >= 0

        self.epsilon = epsilon
        self.variables = variables

    def fit(self, X, y):
        """Calculate the WOE for every column.

        Args:
            X (np.array): (binned) features
            y (np.array): target
        """
        assert y is not None, "WoEBucketer needs a target y"
        assert len(np.unique(y)) == 2, "WoEBucketer is only suited for binary classification"

        X = self._is_dataframe(X)
        self._check_contains_na(X, self.variables)

        self.features_bucket_mapping_ = {}

        for var in self.variables:
            bins, woe, _, _ = woe_1d(X[var], y, epsilon=self.epsilon)

            self.features_bucket_mapping_[var] = BucketMapping(
                feature_name=var, type="categorical", map=dict(zip(bins, woe))
            )

        return self

    def transform(self, X):
        """Transform X."""
        return super().transform(X)
