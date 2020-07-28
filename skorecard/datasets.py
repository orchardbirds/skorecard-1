import pkgutil
import io
import pandas as pd


def load_uci_credit_card(return_X_y=False, as_frame=False):
    """Loads the UCI Credit Card Dataset.

    Args:
        return_X_y:  (bool) If True, returns ``(data, target)`` instead of a dict object.
        as_frame: (bool) give the pandas dataframe instead of X, y matrices (default=False).

    Returns: (pd.DataFrame, dict or tuple) features and target, with as follows:
        - if as_frame is True: returns pd.DataFrame with y as a target
        - return_X_y is True: returns a tuple: (X,y)
        - is both are false (default setting): returns a dictionary where the key `data` contains the features,
        and the key `target` is the target

    """
    file = pkgutil.get_data("skorecard", "data/UCI_Credit_Card.zip")
    df = pd.read_csv(io.BytesIO(file), compression="zip")
    df = df.rename(columns={"default.payment.next.month": "default"})
    if as_frame:
        return df[["EDUCATION", "MARRIAGE", "LIMIT_BAL", "BILL_AMT1", "default"]]
    X, y = (
        df[["EDUCATION", "MARRIAGE", "LIMIT_BAL", "BILL_AMT1"]].values,
        df["default"].values,
    )
    if return_X_y:
        return X, y

    return {"data": X, "target": y}
