from skorecard import datasets
from skorecard.bucketers import DecisionTreeBucketer
from skorecard.reporting import create_report
import numpy as np
import pandas as pd

import pytest


@pytest.fixture()
def df():
    """Generate dataframe."""
    return datasets.load_uci_credit_card(as_frame=True)


def test_report_decision_tree(df):
    """Test the reporting module."""
    X = df[["LIMIT_BAL", "BILL_AMT1"]]
    y = df["default"]
    tbt = DecisionTreeBucketer(max_n_bins=4, min_bin_size=0.1, variables=["LIMIT_BAL", "BILL_AMT1"])
    tbt.fit(X, y)
    tbt.transform(X)

    df_out = create_report(X, y, column="LIMIT_BAL", bucketer=tbt)
    assert df_out.shape == (4, 11)
    assert df_out["bin_labels"].to_dict() == tbt.features_bucket_mapping_["LIMIT_BAL"].labels

    expected = pd.DataFrame({"Bucket_id": {0: 0, 1: 1, 2: 2, 3: 3}, "Count": {0: 849, 1: 676, 2: 1551, 3: 2924}})
    pd.testing.assert_frame_equal(df_out[["Bucket_id", "Count"]], expected)

    np.testing.assert_array_equal(
        df_out.columns.ravel(),
        np.array(
            [
                "Bucket_id",
                "bin_labels",
                "Count",
                "Count (%)",
                "Event",
                "% Event",
                "Non Event",
                "% Non Event",
                "Event Rate",
                "WoE",
                "IV",
            ]
        ),
    )
