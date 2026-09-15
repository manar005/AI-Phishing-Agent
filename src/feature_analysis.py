# Analyze cleaned datasets to decide which features to use for phishing detection.

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MEAJOR_CLEANED = PROCESSED_DIR / "meajor_cleaned.csv"
KAGGLE_CLEANED = PROCESSED_DIR / "kaggle_cleaned.csv"

TARGET_COLUMN = "label"
LOW_CARDINALITY_MAX = 20
IDENTIFIER_UNIQUE_RATIO = 0.30

TEXT_COLUMNS = {
    "raw_text",
    "subject",
    "body",
    "body_plain",
    "body_html",
    "urls",
}

LEAKAGE_COLUMNS = {
    "x_spam_score",
    "source",
}

IDENTIFIER_NAME_HINTS = {
    "message_id",
    "in_reply_to",
    "received_origin_ip",
    "from_address",
    "to_addresses",
    "cc_addresses",
    "reply_to",
    "list_unsubscribe",
    "sender",
    "receiver",
    "date",
}


def _print_header(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(title)
    print("=" * 72)


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _missing_pct(series: pd.Series) -> float:
    missing = series.isna()
    if series.dtype == object or str(series.dtype).startswith("string"):
        empty = series.astype("string").fillna("").str.strip() == ""
        missing = missing | empty
    return float(missing.mean() * 100)


def _is_numeric_or_boolean(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)


def _to_numeric_feature(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    return pd.to_numeric(series, errors="coerce")


def column_overview(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        nunique = df[column].nunique(dropna=False)
        rows.append(
            {
                "column": column,
                "dtype": str(df[column].dtype),
                "nunique": nunique,
                "unique_ratio": round(nunique / len(df), 4) if len(df) else 0.0,
                "missing_pct": round(_missing_pct(df[column]), 2),
            }
        )
    return pd.DataFrame(rows)


def print_column_overview(overview: pd.DataFrame) -> None:
    _print_header("Columns, dtypes, unique values, missing %")
    print(overview.to_string(index=False))


def print_numeric_boolean_by_label(df: pd.DataFrame) -> None:
    _print_header("Numeric / boolean distributions by label")
    feature_cols = [
        column
        for column in df.columns
        if column != TARGET_COLUMN and _is_numeric_or_boolean(df[column])
    ]
    if not feature_cols:
        print("(none)")
        return

    for column in feature_cols:
        print(f"\n{column}")
        feature = _to_numeric_feature(df[column])
        grouped = (
            pd.DataFrame({column: feature, TARGET_COLUMN: df[TARGET_COLUMN]})
            .groupby(TARGET_COLUMN)[column]
            .agg(["count", "mean", "std", "min", "median", "max"])
            .round(4)
        )
        print(grouped.to_string())


def print_low_cardinality_crosstabs(df: pd.DataFrame, overview: pd.DataFrame) -> None:
    _print_header(f"Low-cardinality categorical crosstabs (nunique <= {LOW_CARDINALITY_MAX})")
    candidates = overview[
        (overview["column"] != TARGET_COLUMN)
        & (overview["nunique"] <= LOW_CARDINALITY_MAX)
        & (~overview["column"].isin(TEXT_COLUMNS))
    ]["column"].tolist()
    categorical = [
        column
        for column in candidates
        if not pd.api.types.is_numeric_dtype(df[column]) or pd.api.types.is_bool_dtype(df[column])
    ]
    if not categorical:
        print("(none)")
        return

    for column in categorical:
        print(f"\n{column}")
        counts = pd.crosstab(df[column].fillna("(missing)"), df[TARGET_COLUMN], margins=True)
        percents = pd.crosstab(
            df[column].fillna("(missing)"),
            df[TARGET_COLUMN],
            normalize="index",
        ).mul(100).round(2)
        print("counts:")
        print(counts.to_string())
        print("row % by label:")
        print(percents.to_string())


def print_label_correlations(df: pd.DataFrame) -> pd.Series:
    _print_header("Correlation with label (numeric / boolean)")
    correlations = {}
    label = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    for column in df.columns:
        if column == TARGET_COLUMN or not _is_numeric_or_boolean(df[column]):
            continue
        feature = _to_numeric_feature(df[column])
        correlations[column] = feature.corr(label)

    corr_series = pd.Series(correlations, dtype="float64").dropna().sort_values(key=abs, ascending=False)
    if corr_series.empty:
        print("(none)")
        return corr_series
    print(corr_series.round(4).to_string())
    return corr_series


def identifier_like_columns(overview: pd.DataFrame) -> list[str]:
    flagged = []
    for row in overview.itertuples(index=False):
        column = row.column
        if column == TARGET_COLUMN or column in TEXT_COLUMNS:
            continue
        name = column.lower()
        high_cardinality = row.unique_ratio >= IDENTIFIER_UNIQUE_RATIO or (
            row.nunique >= 1000 and row.unique_ratio >= 0.10
        )
        name_is_identifier = (
            column in IDENTIFIER_NAME_HINTS or name.endswith("_id") or name.endswith("_ip")
        )
        if high_cardinality or name_is_identifier:
            flagged.append(column)
    return flagged


def print_identifier_like_columns(overview: pd.DataFrame, identifiers: list[str]) -> None:
    _print_header("High-cardinality / identifier-like columns")
    if not identifiers:
        print("(none)")
        return
    subset = overview[overview["column"].isin(identifiers)][
        ["column", "dtype", "nunique", "unique_ratio", "missing_pct"]
    ]
    print(subset.to_string(index=False))
    print("\nThese look like IDs, addresses, timestamps, or near-unique fields and are risky as model features.")


def group_columns(df: pd.DataFrame, overview: pd.DataFrame, identifiers: list[str]) -> dict[str, list[str]]:
    identifiers_set = set(identifiers) | (LEAKAGE_COLUMNS & set(df.columns))
    groups = {
        "likely useful features": [],
        "possible identifier/leakage features": [],
        "target column": [],
        "text columns": [],
    }
    for column in df.columns:
        if column == TARGET_COLUMN:
            groups["target column"].append(column)
        elif column in TEXT_COLUMNS:
            groups["text columns"].append(column)
        elif column in identifiers_set:
            groups["possible identifier/leakage features"].append(column)
        else:
            groups["likely useful features"].append(column)
    return groups


def print_feature_groups(groups: dict[str, list[str]]) -> None:
    _print_header("Feature grouping summary")
    for group_name, columns in groups.items():
        print(f"\n{group_name}:")
        if not columns:
            print("  (none)")
            continue
        for column in columns:
            print(f"  - {column}")


def analyze_dataset(path: Path, dataset_name: str) -> None:
    df = load_dataset(path)
    _print_header(f"{dataset_name}: {path}")
    print(f"rows: {len(df):,}")
    print(f"columns: {len(df.columns)}")

    overview = column_overview(df)
    print_column_overview(overview)
    print_numeric_boolean_by_label(df)
    print_low_cardinality_crosstabs(df, overview)
    print_label_correlations(df)

    identifiers = identifier_like_columns(overview)
    print_identifier_like_columns(overview, identifiers)
    groups = group_columns(df, overview, identifiers)
    print_feature_groups(groups)


def main() -> None:
    datasets = (
        ("MeAJOR", MEAJOR_CLEANED),
        ("Kaggle", KAGGLE_CLEANED),
    )
    for dataset_name, path in datasets:
        if not path.exists():
            raise SystemExit(f"Cleaned dataset not found: {path}")
        analyze_dataset(path, dataset_name)


if __name__ == "__main__":
    main()
