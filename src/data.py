from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, FEATURES, PROCESSED_PATH, RANDOM_STATE, TARGET, TEST_SIZE


def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python download_data.py` first."
        )
    df = pd.read_csv(path)
    return df

def _normalize_target_column(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame[TARGET] = frame[TARGET].astype(int)
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        frame[TARGET] = frame[TARGET].apply(lambda value: 1 if value > 0 else 0)
    return frame


def prepare_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    missing_columns = [col for col in FEATURES + [TARGET] if col not in cleaned.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    cleaned = cleaned.replace(["?", "NA", "N/A", ""], pd.NA)
    for column in FEATURES + [TARGET]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=[TARGET])
    cleaned = _normalize_target_column(cleaned)
    return cleaned[FEATURES + [TARGET]]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    return prepare_dataset(df)


def split_data(df: pd.DataFrame):
    features = df[FEATURES]
    labels = df[TARGET]
    return train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

def save_clean_data(df: pd.DataFrame):
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    return PROCESSED_PATH
