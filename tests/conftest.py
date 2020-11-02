import random
import pytest

from skorecard import datasets


@pytest.fixture()
def df():
    """Generate dataframe."""
    df = datasets.load_uci_credit_card(as_frame=True)
    # Add a fake categorical
    pets = ["no pets"] * 3000 + ["cat lover"] * 1500 + ["dog lover"] * 1000 + ["rabbit"] * 498 + ["gold fish"] * 2
    random.Random(42).shuffle(pets)
    df["pet_ownership"] = pets
    return df