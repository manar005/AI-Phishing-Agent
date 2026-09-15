# Train a baseline logistic regression model on the cleaned MeAJOR dataset.

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
    MEAJOR_EXCLUDED_FEATURES,
    MEAJOR_STRUCTURAL_FEATURES,
    MEAJOR_TEXT_FEATURES,
    TARGET_COLUMN,
)

MEAJOR_CLEANED = PROJECT_ROOT / "data" / "processed" / "meajor_cleaned.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20

CATEGORICAL_FEATURES = ["content_types"]
BOOLEAN_FEATURES = ["has_attachments"]
NUMERIC_FEATURES = [
    column
    for column in MEAJOR_STRUCTURAL_FEATURES
    if column not in CATEGORICAL_FEATURES and column not in BOOLEAN_FEATURES
]


def combine_subject_and_body(values) -> np.ndarray:
    """Concatenate subject and body into a single document per row."""
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
    feature_columns = MEAJOR_TEXT_FEATURES + MEAJOR_STRUCTURAL_FEATURES
    leaked = set(feature_columns) & set(MEAJOR_EXCLUDED_FEATURES)
    if leaked:
        raise ValueError(f"Excluded/leakage columns present in the feature list: {sorted(leaked)}")

    missing = [column for column in feature_columns + [TARGET_COLUMN] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df.loc[:, feature_columns].copy()
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def build_pipeline() -> Pipeline:
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
            ("text", text_pipeline, MEAJOR_TEXT_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("boolean", boolean_pipeline, BOOLEAN_FEATURES),
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


def main() -> None:
    if not MEAJOR_CLEANED.exists():
        raise SystemExit(f"Cleaned dataset not found: {MEAJOR_CLEANED}")

    X, y = load_and_prepare_data(MEAJOR_CLEANED)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("MeAJOR baseline logistic regression")
    print(f"training-set size: {len(X_train):,}")
    print(f"test-set size:     {len(X_test):,}")

    pipeline = build_pipeline()
    trained = train_model(pipeline, X_train, y_train)
    evaluate_model(trained, X_test, y_test)


if __name__ == "__main__":
    main()
