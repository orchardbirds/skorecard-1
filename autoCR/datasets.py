from pkg_resources import resource_filename
import os
import pandas as pd


def load_uci_credit_card(return_X_y=False, as_frame=False):
    """Loads the UCI Credit Card Dataset.

    :param return_X_y: If True, returns ``(data, target)`` instead of a dict object.
    :param as_frame: give the pandas dataframe instead of X, y matrices (default=False).
    :return: dict with data and target.
    """
    filepath = resource_filename("autoCR", os.path.join("data", "UCI_Credit_Card.zip"))
    df = pd.read_csv(filepath)
    if as_frame:
        return df
    X, y = (
        df[["EDUCATION", "MARRIAGE", "LIMIT_BAL", "BILL_AMT4"]].values,
        df["default.payment.next.month"].values,
    )
    if return_X_y:
        return X, y
    return {"data": X, "target": y}
