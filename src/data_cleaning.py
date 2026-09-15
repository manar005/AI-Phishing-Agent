# Clean MeAJOR and Kaggle datasets into data/processed without modifying the raw CSVs.

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

MEAJOR_RAW = DATA_DIR / "meajor_dataset.csv"
KAGGLE_RAW = DATA_DIR / "kaggle_dataset.csv"
MEAJOR_CLEANED = PROCESSED_DIR / "meajor_cleaned.csv"
KAGGLE_CLEANED = PROCESSED_DIR / "kaggle_cleaned.csv"

KAGGLE_BOOLEAN_COLUMNS = (
    "has_attachments",
    "has_html",
    "contains_tracking_token",
)

_TRUE_VALUES = {"true", "1", "t", "yes", "y"}
_FALSE_VALUES = {"false", "0", "f", "no", "n"}


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def save_dataset(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def convert_label_to_int(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["label"] = pd.to_numeric(cleaned["label"], errors="raise").astype(int)
    return cleaned


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        raise ValueError("Boolean columns contain missing values; leaving them unchanged is not supported.")
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    raise ValueError(f"Cannot standardize {value!r} as a boolean.")


def standardize_boolean_columns(df: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> pd.DataFrame:
    cleaned = df.copy()
    for column in columns:
        if column not in cleaned.columns:
            continue
        cleaned[column] = cleaned[column].map(_to_bool).astype(bool)
    return cleaned


def clean_meajor(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.dropna(subset=["label"])
    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.drop_duplicates(subset=["subject", "body"])
    cleaned = convert_label_to_int(cleaned)
    return cleaned.reset_index(drop=True)


def clean_kaggle(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.drop_duplicates(subset=["subject", "body_plain"])
    cleaned = convert_label_to_int(cleaned)
    cleaned = standardize_boolean_columns(cleaned, KAGGLE_BOOLEAN_COLUMNS)
    return cleaned.reset_index(drop=True)


def _print_cleaning_summary(dataset_name: str, rows_before: int, rows_after: int, output_path: Path) -> None:
    print(f"\n{dataset_name}")
    print(f"  rows before: {rows_before:,}")
    print(f"  rows after:  {rows_after:,}")
    print(f"  saved to:    {output_path}")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    meajor_raw = load_dataset(MEAJOR_RAW)
    meajor_cleaned = clean_meajor(meajor_raw)
    save_dataset(meajor_cleaned, MEAJOR_CLEANED)
    _print_cleaning_summary("MeAJOR", len(meajor_raw), len(meajor_cleaned), MEAJOR_CLEANED)

    kaggle_raw = load_dataset(KAGGLE_RAW)
    kaggle_cleaned = clean_kaggle(kaggle_raw)
    save_dataset(kaggle_cleaned, KAGGLE_CLEANED)
    _print_cleaning_summary("Kaggle", len(kaggle_raw), len(kaggle_cleaned), KAGGLE_CLEANED)


if __name__ == "__main__":
    main()
