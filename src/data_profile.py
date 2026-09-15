# Profile email datasets: shape, missingness, label balance, and feature distributions.

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

LONG_TEXT_COLUMNS = {
    "raw_text",
    "subject",
    "body",
    "body_plain",
    "body_html",
    "urls",
}


def _print_header(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(title)
    print("=" * 72)


def _missing_table(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    empty_strings = pd.Series(0, index=df.columns, dtype="int64")
    for column in df.select_dtypes(include=["object", "string"]).columns:
        empty_strings[column] = (df[column].astype("string").fillna("").str.strip() == "").sum()

    table = pd.DataFrame(
        {
            "missing": missing,
            "empty_string": empty_strings,
            "missing_pct": (missing / len(df) * 100).round(2),
        }
    )
    return table[table[["missing", "empty_string"]].gt(0).any(axis=1)].sort_values(
        "missing",
        ascending=False,
    )


def _duplicate_count(df: pd.DataFrame, subset: list[str] | None = None) -> tuple[int, float]:
    if df.empty:
        return 0, 0.0
    n_duplicates = int(df.duplicated(subset=subset).sum())
    return n_duplicates, n_duplicates / len(df) * 100


def _content_key_columns(df: pd.DataFrame, name: str) -> list[str] | None:
    stem = Path(name).stem.lower()
    if "kaggle" in stem:
        keys = ["subject", "body_plain"]
    elif "meajor" in stem:
        keys = ["subject", "body"]
    elif {"subject", "body_plain"}.issubset(df.columns):
        keys = ["subject", "body_plain"]
    elif {"subject", "body"}.issubset(df.columns):
        keys = ["subject", "body"]
    else:
        return None
    if any(column not in df.columns for column in keys):
        return None
    return keys


def _print_duplicate_analysis(df: pd.DataFrame, name: str) -> None:
    _print_header("Duplicate analysis")
    n_full, pct_full = _duplicate_count(df)
    print(f"completely duplicated rows: {n_full:,} ({pct_full:.2f}%)")

    content_keys = _content_key_columns(df, name)
    if content_keys is None:
        print("duplicate email content: skipped (subject/body columns not found)")
        return

    n_content, pct_content = _duplicate_count(df, subset=content_keys)
    key_label = " + ".join(content_keys)
    print(f"duplicate email content ({key_label}): {n_content:,} ({pct_content:.2f}%)")


def _text_length_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        if column not in LONG_TEXT_COLUMNS and df[column].dtype not in ("object", "string"):
            continue
        if column not in LONG_TEXT_COLUMNS:
            continue
        lengths = df[column].fillna("").astype(str).str.len()
        rows.append(
            {
                "column": column,
                "mean_len": round(lengths.mean(), 2),
                "median_len": int(lengths.median()),
                "max_len": int(lengths.max()),
                "empty_pct": round((lengths == 0).mean() * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def profile_dataframe(df: pd.DataFrame, name: str) -> None:
    _print_header(f"Dataset: {name}")
    print(f"rows: {len(df):,}")
    print(f"columns: {len(df.columns)}")
    print(f"memory: {df.memory_usage(deep=True).sum() / 1_048_576:.1f} MB")

    _print_header("Columns and dtypes")
    print(df.dtypes.to_string())

    _print_header("Missing values")
    missing = _missing_table(df)
    print("(none)" if missing.empty else missing.to_string())

    _print_duplicate_analysis(df, name)

    if "label" in df.columns:
        _print_header("Label distribution")
        counts = df["label"].value_counts(dropna=False).sort_index()
        percents = (counts / counts.sum() * 100).round(2)
        print(pd.DataFrame({"count": counts, "percent": percents}).to_string())

    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    if numeric_cols:
        _print_header("Numeric / boolean summary")
        print(df[numeric_cols].describe(include="all").transpose().to_string())

    _print_header("Categorical value counts")
    categorical_cols = [
        column
        for column in df.select_dtypes(include=["object", "string", "bool"]).columns
        if column not in LONG_TEXT_COLUMNS and column != "label"
    ]
    if not categorical_cols:
        print("(none)")
    else:
        identifier_threshold = max(50, int(len(df) * 0.1))
        for column in categorical_cols:
            nunique = df[column].nunique(dropna=False)
            print(f"\n{column} (unique={nunique:,})")
            if nunique > identifier_threshold:
                print("high-cardinality identifier; skipped value counts")
            elif nunique > 20:
                print(df[column].value_counts(dropna=False).head(10).to_string())
                print("...")
            else:
                print(df[column].value_counts(dropna=False).to_string())

    text_stats = _text_length_stats(df)
    if not text_stats.empty:
        _print_header("Text length stats")
        print(text_stats.to_string(index=False))


def load_dataset(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile phishing email CSV datasets.")
    parser.add_argument(
        "--path",
        type=Path,
        default=DATA_DIR / "kaggle_dataset.csv",
        help="CSV file to profile. Defaults to data/kaggle_dataset.csv.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Profile every CSV in data/.",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional row cap for a faster sample profile.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(DATA_DIR.glob("*.csv")) if args.all else [args.path]
    if not paths:
        raise SystemExit(f"No CSV files found in {DATA_DIR}")

    for path in paths:
        resolved = path if path.is_absolute() else PROJECT_ROOT / path
        if not resolved.exists():
            raise SystemExit(f"File not found: {resolved}")
        df = load_dataset(resolved, nrows=args.nrows)
        profile_dataframe(df, resolved.name)


if __name__ == "__main__":
    main()
