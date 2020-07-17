from pkg_resources import resource_filename
import os
import pandas as pd


def load_uci_credit_card(return_X_y=False, as_frame=False):
    """Loads the UCI Credit Card Dataset.

    Args:
        return_X_y:  (bool) If True, returns ``(data, target)`` instead of a dict object.
        as_frame: (bool) give the pandas dataframe instead of X, y matrices (default=False).

    Returns: data and target as dictionary if return_X_y is True or pandas dataframe if as_frame is True.

    """
    filepath = resource_filename("autoCR", os.path.join("data", "UCI_Credit_Card.zip"))
    df = pd.read_csv(filepath)
    df = df.rename(columns={"default.payment.next.month": "default"})
    if as_frame:
        return df
    X, y = (
        df[["EDUCATION", "MARRIAGE", "LIMIT_BAL", "BILL_AMT1"]].values,
        df["default"].values,
    )
    if return_X_y:
        return X, y

    return {"data": X, "target": y}
