# Train a baseline logistic regression model on the cleaned Kaggle dataset.
#
# This project prioritizes generalization to unseen phishing emails over
# maximizing accuracy on a particular dataset. Dataset-specific identities
# and synthetic shortcut features can inflate scores without capturing real
# phishing behavior, so they are kept out of the detector's target design.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_config import (  # noqa: E402
    KAGGLE_EXCLUDED_FEATURES,
    KAGGLE_SECURITY_FEATURES,
    KAGGLE_TEXT_FEATURES,
    TARGET_COLUMN,
)

KAGGLE_CLEANED = PROJECT_ROOT / "data" / "processed" / "kaggle_cleaned.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Original baseline feature groups. Unchanged from the first Kaggle experiment.
CATEGORICAL_FEATURES = ["spf_result", "dkim_result", "dmarc_result"]
BOOLEAN_FEATURES = ["has_attachments", "has_html", "contains_tracking_token"]
NUMERIC_FEATURES = [
    column
    for column in KAGGLE_SECURITY_FEATURES
    if column not in CATEGORICAL_FEATURES and column not in BOOLEAN_FEATURES
]
BASELINE_FEATURES = KAGGLE_TEXT_FEATURES + KAGGLE_SECURITY_FEATURES

# Behavior-focused ablation: drop synthetic shortcut counts/flags that can
# separate this dataset without reflecting general phishing behavior.
ABLATION_REMOVED_FEATURES = [
    "num_urls",
    "num_phone_numbers",
    "contains_tracking_token",
]
ABLATION_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES
ABLATION_BOOLEAN_FEATURES = ["has_attachments", "has_html"]
ABLATION_NUMERIC_FEATURES = [
    column for column in NUMERIC_FEATURES if column not in ABLATION_REMOVED_FEATURES
]
ABLATION_FEATURES = [column for column in BASELINE_FEATURES if column not in ABLATION_REMOVED_FEATURES]


def combine_subject_and_body(values) -> np.ndarray:
    """Concatenate subject and body_plain into a single document per row."""
    if isinstance(values, pd.DataFrame):
        subject = values.iloc[:, 0]
        body = values.iloc[:, 1]
    else:
        array = np.asarray(values, dtype=object)
        subject = pd.Series(array[:, 0])
        body = pd.Series(array[:, 1])
    combined = subject.fillna("").astype(str) + "\n" + body.fillna("").astype(str)
    return combined.to_numpy()


def boolean_to_float(values) -> np.ndarray:
    """Convert boolean columns to float so missing values can be imputed."""
    return pd.DataFrame(values).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def load_and_prepare_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, low_memory=False)
    leaked = set(BASELINE_FEATURES) & set(KAGGLE_EXCLUDED_FEATURES)
    if leaked:
        raise ValueError(f"Excluded/leakage columns present in the feature list: {sorted(leaked)}")

    missing = [column for column in BASELINE_FEATURES + [TARGET_COLUMN] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df.loc[:, BASELINE_FEATURES].copy()
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def build_pipeline(
    text_features: list[str],
    categorical_features: list[str],
    numeric_features: list[str],
    boolean_features: list[str],
) -> Pipeline:
    text_pipeline = Pipeline(
        steps=[
            ("combine", FunctionTransformer(combine_subject_and_body, validate=False)),
            ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1, 1))),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    boolean_pipeline = Pipeline(
        steps=[
            ("to_float", FunctionTransformer(boolean_to_float, validate=False)),
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_pipeline, text_features),
            ("categorical", categorical_pipeline, categorical_features),
            ("numeric", numeric_pipeline, numeric_features),
            ("boolean", boolean_pipeline, boolean_features),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
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


def train_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\nTest label distribution:")
    print(y_test.value_counts(normalize=False).sort_index().to_string())
    print("\nTest label distribution (%):")
    print((y_test.value_counts(normalize=True).sort_index() * 100).round(2).to_string())

    print(f"\naccuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"precision: {precision_score(y_test, y_pred):.4f}")
    print(f"recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"f1:        {f1_score(y_test, y_pred):.4f}")
    print(f"roc-auc:   {roc_auc_score(y_test, y_proba):.4f}")

    print("\nconfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nclassification report:")
    print(classification_report(y_test, y_pred, digits=4))


def run_experiment(
    title: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    text_features: list[str],
    categorical_features: list[str],
    numeric_features: list[str],
    boolean_features: list[str],
) -> None:
    feature_columns = text_features + categorical_features + numeric_features + boolean_features
    print(f"\n{'=' * 72}")
    print(title)
    print("=" * 72)
    print(f"training-set size: {len(X_train):,}")
    print(f"test-set size:     {len(X_test):,}")
    print(f"features:          {feature_columns}")

    pipeline = build_pipeline(
        text_features,
        categorical_features,
        numeric_features,
        boolean_features,
    )
    trained = train_model(pipeline, X_train[feature_columns], y_train)
    evaluate_model(trained, X_test[feature_columns], y_test)


def main() -> None:
    if not KAGGLE_CLEANED.exists():
        raise SystemExit(f"Cleaned dataset not found: {KAGGLE_CLEANED}")

    X, y = load_and_prepare_data(KAGGLE_CLEANED)
    # One shared split so baseline and ablation scores are comparable.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    run_experiment(
        "Kaggle baseline logistic regression",
        X_train,
        X_test,
        y_train,
        y_test,
        KAGGLE_TEXT_FEATURES,
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
        BOOLEAN_FEATURES,
    )
    run_experiment(
        "Kaggle behavior-focused ablation (shortcut features removed)",
        X_train,
        X_test,
        y_train,
        y_test,
        KAGGLE_TEXT_FEATURES,
        ABLATION_CATEGORICAL_FEATURES,
        ABLATION_NUMERIC_FEATURES,
        ABLATION_BOOLEAN_FEATURES,
    )


if __name__ == "__main__":
    main()
