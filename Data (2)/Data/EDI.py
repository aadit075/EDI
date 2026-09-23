"""Independent per-hospital cardiac-risk model training pipeline."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


DATA_DIR: Final[Path] = Path(__file__).resolve().parent
OUTPUT_DIR: Final[Path] = DATA_DIR / "outputs_independent"
HOSPITALS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E")
TARGET: Final[str] = "cardiac_risk_label"
ID_COLUMNS: Final[tuple[str, ...]] = ("patient_id", "hospital_id")
RANDOM_STATE: Final[int] = 42

NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "age", "height_cm", "weight_kg", "bmi", "heart_rate",
    "systolic_bp", "diastolic_bp", "oxygen_saturation",
    "respiratory_rate", "body_temperature", "diabetes", "hypertension",
    "coronary_artery_disease", "family_history_cardiac_disease",
    "chest_pain", "shortness_of_breath", "fatigue", "palpitations",
    "hemoglobin", "white_blood_cell_count", "creatinine", "blood_glucose",
    "hba1c", "total_cholesterol", "ldl_cholesterol", "hdl_cholesterol",
    "triglycerides", "sodium", "potassium", "cardiac_biomarker",
    "previous_cardiac_hospitalization",
)
CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = (
    "sex", "smoking_status", "physical_activity_level",
)
FEATURE_COLUMNS: Final[tuple[str, ...]] = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS




def json_safe(value: Any) -> Any:
    """Convert NumPy and pandas values into JSON-compatible values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=json_safe)


def one_hot_encoder() -> OneHotEncoder:
    """Create a version-compatible one-hot encoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def read_split(hospital_id: str, split: str) -> pd.DataFrame:
    """Read and normalize one hospital split."""
    if hospital_id not in HOSPITALS:
        raise ValueError(f"Unknown hospital: {hospital_id}")
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Invalid split: {split}")
    path = DATA_DIR / f"hospital_{hospital_id}_{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset: {path}")
    frame = pd.read_csv(path, na_values=["", "NA", "N/A", "null", "None"])
    required = set(FEATURE_COLUMNS) | set(ID_COLUMNS) | {TARGET}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    frame = frame.copy()
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in CATEGORICAL_COLUMNS:
        frame[column] = frame[column].astype(object)
        frame[column] = frame[column].where(frame[column].notna(), np.nan)
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    if frame[TARGET].isna().any() or not frame[TARGET].isin([0, 1]).all():
        raise ValueError(f"{path.name} contains an invalid target value")
    return frame


def validate_splits(hospital_id: str) -> None:
    """Validate schema, IDs, labels, and split separation for one hospital."""
    frames = {split: read_split(hospital_id, split) for split in ("train", "validation", "test")}
    expected_columns = set(frames["train"].columns)
    for split, frame in frames.items():
        if set(frame.columns) != expected_columns:
            raise ValueError(f"Column mismatch in Hospital {hospital_id} {split}")
        if frame["patient_id"].isna().any() or frame["patient_id"].duplicated().any():
            raise ValueError(f"Invalid or duplicate patient IDs in Hospital {hospital_id} {split}")
        if not frame["hospital_id"].eq(f"Hospital_{hospital_id}").all():
            raise ValueError(f"Incorrect hospital_id in Hospital {hospital_id} {split}")
    train_ids = set(frames["train"]["patient_id"])
    validation_ids = set(frames["validation"]["patient_id"])
    test_ids = set(frames["test"]["patient_id"])
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise ValueError(f"Patient ID overlap detected for Hospital {hospital_id}")


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate approved features and the binary target."""
    return frame.loc[:, list(FEATURE_COLUMNS)].copy(), frame[TARGET].astype(int).copy()


def build_preprocessor() -> ColumnTransformer:
    """Build a hospital-local preprocessing transformer for XGBoost."""
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(NUMERIC_COLUMNS)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_COLUMNS)),
        ],
        remainder="drop",
    )


def build_xgboost_model(positive_weight: float) -> Pipeline:
    """Build the single XGBoost pipeline used for every hospital."""
    classifier = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=positive_weight,
    )
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", classifier),
        ]
    )


def calculate_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    """Calculate classification metrics for a probability threshold."""
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    metrics: dict[str, float] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "true_negative": float(matrix[0, 0]),
        "false_positive": float(matrix[0, 1]),
        "false_negative": float(matrix[1, 0]),
        "true_positive": float(matrix[1, 1]),
    }
    metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities)) if y_true.nunique() == 2 else float("nan")
    return metrics


def select_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Select the validation threshold maximizing F1, then recall, then lower threshold."""
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    best_threshold = 0.5
    best_score = (-1.0, -1.0, float("-inf"))
    for candidate in candidates:
        metrics = calculate_metrics(y_true, probabilities, float(candidate))
        score = (metrics["f1_score"], metrics["recall"], -float(candidate))
        if score > best_score:
            best_score = score
            best_threshold = float(candidate)
    return best_threshold


def run_eda(hospital_id: str, train: pd.DataFrame, output_dir: Path) -> None:
    """Write training-only EDA summaries for one hospital."""
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train = split_features_target(train)
    missing = x_train.isna().sum().rename("missing_count").to_frame()
    missing["missing_percent"] = missing["missing_count"] / len(x_train) * 100.0
    missing.to_csv(output_dir / "missingness.csv")
    numeric_summary = x_train[list(NUMERIC_COLUMNS)].describe().T
    numeric_summary.to_csv(output_dir / "numeric_summary.csv")
    categorical_rows: list[dict[str, Any]] = []
    for column in CATEGORICAL_COLUMNS:
        for category, count in x_train[column].value_counts(dropna=False).items():
            mask = x_train[column].eq(category) if pd.notna(category) else x_train[column].isna()
            categorical_rows.append({
                "feature": column,
                "category": "<MISSING>" if pd.isna(category) else str(category),
                "count": int(count),
                "positive_rate": float(y_train[mask].mean()) if mask.any() else float("nan"),
            })
    pd.DataFrame(categorical_rows).to_csv(output_dir / "categorical_summary.csv", index=False)
    correlations = x_train[list(NUMERIC_COLUMNS)].corrwith(y_train).sort_values(key=lambda values: values.abs(), ascending=False)
    correlations.rename("correlation_with_target").to_csv(output_dir / "numeric_target_correlations.csv")
    write_json(output_dir / "eda_summary.json", {
        "hospital_id": hospital_id,
        "rows": int(len(train)),
        "feature_count": len(FEATURE_COLUMNS),
        "positive_count": int(y_train.sum()),
        "negative_count": int((1 - y_train).sum()),
        "positive_rate": float(y_train.mean()),
        "feature_columns": list(FEATURE_COLUMNS),
        "numeric_columns": list(NUMERIC_COLUMNS),
        "categorical_columns": list(CATEGORICAL_COLUMNS),
    })


def train_hospital(hospital_id: str) -> dict[str, Any]:
    """Train and test one independent XGBoost model for one hospital."""
    validate_splits(hospital_id)
    hospital_output = OUTPUT_DIR / f"hospital_{hospital_id}"
    hospital_output.mkdir(parents=True, exist_ok=True)
    train = read_split(hospital_id, "train")
    validation = read_split(hospital_id, "validation")
    test = read_split(hospital_id, "test")
    run_eda(hospital_id, train, hospital_output / "eda")
    x_train, y_train = split_features_target(train)
    x_validation, y_validation = split_features_target(validation)
    x_test, y_test = split_features_target(test)
    positive_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    model = build_xgboost_model(positive_weight)
    model.fit(x_train, y_train)
    validation_probabilities = model.predict_proba(x_validation)[:, 1]
    threshold = select_threshold(y_validation, validation_probabilities)
    validation_metrics = calculate_metrics(y_validation, validation_probabilities, threshold)
    test_probabilities = model.predict_proba(x_test)[:, 1]
    test_metrics = calculate_metrics(y_test, test_probabilities, threshold)
    joblib.dump(model, hospital_output / "xgboost_pipeline.joblib")
    pd.DataFrame([validation_metrics]).to_csv(hospital_output / "validation_metrics.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(hospital_output / "test_metrics.csv", index=False)
    pd.DataFrame({
        "patient_id": test["patient_id"],
        "actual": y_test,
        "probability": test_probabilities,
        "prediction": (test_probabilities >= threshold).astype(int),
    }).to_csv(hospital_output / "test_predictions.csv", index=False)
    write_json(hospital_output / "model_metadata.json", {
        "hospital_id": hospital_id,
        "model": "xgboost",
        "threshold": threshold,
        "positive_weight": positive_weight,
        "training_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "features": list(FEATURE_COLUMNS),
        "data_policy": "Only this hospital's train split fitted preprocessing and XGBoost; validation selected the threshold; test was used for final evaluation.",
    })
    return {"hospital_id": hospital_id, "model": "xgboost", **test_metrics}


def main() -> None:
    """Run independent training and final evaluation for all hospitals."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [train_hospital(hospital_id) for hospital_id in HOSPITALS]
    summary = pd.DataFrame(results)
    summary.to_csv(OUTPUT_DIR / "hospital_test_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nArtifacts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
