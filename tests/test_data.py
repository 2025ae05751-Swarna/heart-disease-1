import pandas as pd

from src.config import FEATURES, TARGET
from src.data import clean_data


def test_target_is_binarized_and_features_are_retained():
    data = {column: [1, 2, 3] for column in FEATURES}
    data[TARGET] = [0, 1, 4]
    table = clean_data(pd.DataFrame(data))
    assert list(table.columns) == FEATURES + [TARGET]
    assert set(table[TARGET].unique()) == {0, 1}


def test_rows_with_missing_target_are_removed():
    data = {column: [1, 2] for column in FEATURES}
    data[TARGET] = [0, None]
    table = clean_data(pd.DataFrame(data))
    assert len(table) == 1
