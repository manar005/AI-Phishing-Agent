# Leave-one-source-out evaluation for the MeAJOR detector.
#
# `source` is never a model input. It is used only to hold out an entire
# email corpus so we can test whether the detector generalizes to unseen
# phishing/legitimate mail rather than memorizing one TREC collection.
#
# Three feature configurations are compared under the same source splits:
# text-only, structural-only, and the combined baseline feature set.

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_config import (  # noqa: E402
    MEAJOR_EXCLUDED_FEATURES,
    MEAJOR_STRUCTURAL_FEATURES,
    MEAJOR_TEXT_FEATURES,
    TARGET_COLUMN,
)
from src.train_meajor import (  # noqa: E402
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    MEAJOR_CLEANED,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    boolean_to_float,
    combine_subject_and_body,
    train_model,
)

SOURCE_COLUMN = "source"
FEATURE_COLUMNS = MEAJOR_TEXT_FEATURES + MEAJOR_STRUCTURAL_FEATURES

LEAVE_ONE_SOURCE_OUT = (
    (("trec5", "trec6"), "trec7"),
    (("trec5", "trec7"), "trec6"),
    (("trec6", "trec7"), "trec5"),
)

FEATURE_CONFIGS = (
    {
        "name": "text-only",
        "text": MEAJOR_TEXT_FEATURES,
        "categorical": [],
        "numeric": [],
        "boolean": [],
    },
    {
        "name": "structural-only",
        "text": [],
        "categorical": CATEGORICAL_FEATURES,
        "numeric": NUMERIC_FEATURES,
        "boolean": BOOLEAN_FEATURES,
    },
    {
        "name": "combined",
        "text": MEAJOR_TEXT_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
        "numeric": NUMERIC_FEATURES,
        "boolean": BOOLEAN_FEATURES,
    },
)


def load_meajor_with_source(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if "source" in FEATURE_COLUMNS or "source" not in MEAJOR_EXCLUDED_FEATURES:
        raise ValueError("source must stay excluded from MeAJOR model features.")

    df = pd.read_csv(path, low_memory=False)
    required = FEATURE_COLUMNS + [TARGET_COLUMN, SOURCE_COLUMN]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    source = df[SOURCE_COLUMN].astype("string").str.strip().str.lower()
    known_source = source.isin(["trec5", "trec6", "trec7"])
    X = df.loc[known_source, FEATURE_COLUMNS].copy()
    y = df.loc[known_source, TARGET_COLUMN].astype(int)
    source = source.loc[known_source]
    return X, y, source


def build_pipeline_for_config(config: dict) -> Pipeline:
    """Reuse the MeAJOR baseline transformers, including only selected feature groups."""
    transformers = []
    if config["text"]:
        transformers.append(
            (
                "text",
                Pipeline(
                    steps=[
                        ("combine", FunctionTransformer(combine_subject_and_body, validate=False)),
                        ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1, 1))),
                    ]
                ),
                config["text"],
            )
        )
    if config["categorical"]:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                config["categorical"],
            )
        )
    if config["numeric"]:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                config["numeric"],
            )
        )
    if config["boolean"]:
        transformers.append(
            (
                "boolean",
                Pipeline(
                    steps=[
                        ("to_float", FunctionTransformer(boolean_to_float, validate=False)),
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                    ]
                ),
                config["boolean"],
            )
        )
    if not transformers:
        raise ValueError(f"No features selected for configuration {config['name']!r}.")

    return Pipeline(
        steps=[
            ("preprocess", ColumnTransformer(transformers=transformers, remainder="drop")),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def config_feature_columns(config: dict) -> list[str]:
    return config["text"] + config["categorical"] + config["numeric"] + config["boolean"]


def _print_label_distribution(name: str, labels: pd.Series) -> None:
    counts = labels.value_counts(normalize=False).sort_index()
    percents = (labels.value_counts(normalize=True).sort_index() * 100).round(2)
    print(f"\n{name} label distribution:")
    print(counts.to_string())
    print(f"{name} label distribution (%):")
    print(percents.to_string())


def evaluate_split(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    print(f"\naccuracy:  {metrics['accuracy']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall:    {metrics['recall']:.4f}")
    print(f"f1:        {metrics['f1']:.4f}")
    print(f"roc-auc:   {metrics['roc_auc']:.4f}")
    print("\nconfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    return metrics


def run_experiment(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    train_sources: tuple[str, str],
    test_source: str,
    config: dict,
) -> dict:
    feature_columns = config_feature_columns(config)
    print(f"\n{'-' * 72}")
    print(f"configuration:       {config['name']}")
    print(f"features:            {feature_columns}")

    pipeline = train_model(
        build_pipeline_for_config(config),
        X_train[feature_columns],
        y_train,
    )
    metrics = evaluate_split(pipeline, X_test[feature_columns], y_test)
    return {
        "train_sources": "+".join(source.upper() for source in train_sources),
        "test_source": test_source.upper(),
        "configuration": config["name"],
        **metrics,
    }


def print_comparison_table(results: list[dict]) -> None:
    table = pd.DataFrame(results)
    display = table.copy()
    for column in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    print(f"\n{'=' * 72}")
    print("Leave-one-source-out comparison (9 experiments)")
    print("=" * 72)
    print(display.to_string(index=False))


def main() -> None:
    if not MEAJOR_CLEANED.exists():
        raise SystemExit(f"Cleaned dataset not found: {MEAJOR_CLEANED}")

    X, y, source = load_meajor_with_source(MEAJOR_CLEANED)
    results: list[dict] = []

    for train_sources, test_source in LEAVE_ONE_SOURCE_OUT:
        train_mask = source.isin(train_sources)
        test_mask = source == test_source
        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_test, y_test = X.loc[test_mask], y.loc[test_mask]

        print(f"\n{'=' * 72}")
        print("MeAJOR leave-one-source-out")
        print("=" * 72)
        print(f"training sources:    {', '.join(item.upper() for item in train_sources)}")
        print(f"unseen test source:  {test_source.upper()}")
        print(f"training-set size:   {len(X_train):,}")
        print(f"test-set size:       {len(X_test):,}")
        _print_label_distribution("train", y_train)
        _print_label_distribution("test", y_test)

        for config in FEATURE_CONFIGS:
            results.append(
                run_experiment(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    train_sources,
                    test_source,
                    config,
                )
            )

    print_comparison_table(results)


if __name__ == "__main__":
    main()
