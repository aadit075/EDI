"""Interactive FedMedTrust patient intake, storage, and risk prediction.

Flow for one patient:
  1. The user selects a hospital branch.
  2. Patient details are read and validated.
  3. The raw record is saved to that hospital's local SQLite database.
  4. That hospital's saved XGBoost model scores the record.
  5. The prediction is written back to the same database row.

Raw patient data stays in the selected hospital's file. This is an academic
prototype using synthetic data; it is not a medical diagnosis.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from database import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_STORE_DIR,
    HOSPITALS,
    TARGET,
    DatabaseError,
    database_path,
    initialize_database,
    save_patient,
    set_confirmed_outcome,
    update_prediction,
)

VARIANTS: tuple[str, ...] = ("engineered", "onehot")
DEFAULT_VARIANT = "onehot"
CATEGORICAL_DEFAULTS: dict[str, tuple[str, ...]] = {
    "sex": ("Female", "Male", "Other", "Unknown"),
    "smoking_status": ("never", "former", "current", "Unknown"),
    "physical_activity_level": ("low", "moderate", "high", "Unknown"),
}
DISCLAIMER = (
    "DISCLAIMER: This is an academic prototype using synthetic data and is not a "
    "medical diagnosis. Do not use it for patient care, triage, or treatment decisions."
)


class PredictionError(RuntimeError):
    """Raised for artifact, input, or scoring failures."""


def resolve_variant_dir(output_dir: Path = DEFAULT_OUTPUT_DIR, variant: str | None = None) -> Path:
    """Return the artifact directory for one feature-space variant.

    Accepts either outputs/<variant>/ or a directory holding the artifacts
    directly, so an earlier single-variant layout still works.
    """
    base = Path(output_dir)
    if variant is not None and str(variant).strip():
        selected = str(variant).strip().lower()
        if selected not in VARIANTS:
            raise PredictionError(
                f"Unknown variant '{variant}'. Choose one of: {', '.join(VARIANTS)}."
            )
        candidate = base / selected
        if not candidate.is_dir():
            raise PredictionError(
                f"Variant '{selected}' is missing at {candidate}.\n"
                "Run: python retrain_hospitals.py --save-all"
            )
        return candidate
    for candidate in (base / DEFAULT_VARIANT, base):
        if (candidate / "shared_preprocessor.joblib").is_file():
            return candidate
    return base / DEFAULT_VARIANT


# --------------------------------------------------------------------------
# Artifact loading
# --------------------------------------------------------------------------
def _load_joblib(path: Path, description: str) -> Any:
    """Load a joblib artifact, converting failures into readable errors."""
    if not path.is_file():
        raise PredictionError(
            f"Missing {description}: {path}\n"
            "Extract the trained outputs so that outputs/shared_preprocessor.joblib "
            "and outputs/hospital_<A-E>/xgboost_pipeline.joblib exist."
        )
    try:
        return joblib.load(path)
    except Exception as error:  # noqa: BLE001 - surfaced as a clear message
        raise PredictionError(f"Could not load {description} at '{path}': {error}") from error


def _extract_estimator(artifact: Any) -> Any:
    """Return a fitted estimator from a saved artifact or wrapper mapping."""
    if isinstance(artifact, Mapping):
        for key in ("model", "pipeline", "estimator", "classifier"):
            if key in artifact:
                return artifact[key]
    return artifact


def _extract_feature_names(preprocessor: Any) -> list[str]:
    """Read the feature schema exposed by the saved shared preprocessor."""
    names: Any = None
    method = getattr(preprocessor, "feature_names", None)
    if callable(method):
        names = method()
    if names is None:
        names = getattr(preprocessor, "feature_names_", None)
    if names is None and hasattr(preprocessor, "get_feature_names_out"):
        names = preprocessor.get_feature_names_out()
    if names is None:
        raise PredictionError(
            "The saved preprocessor does not expose feature names, so patient input "
            "cannot be checked against the training schema."
        )
    result = [str(name) for name in names]
    if not result:
        raise PredictionError("The saved preprocessor contains an empty feature schema.")
    return result


def load_hospital_model(
    hospital: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    variant: str | None = None,
) -> tuple[Any, Any]:
    """Load the shared preprocessor and one hospital's saved XGBoost model."""
    hospital_id = str(hospital).strip().upper()
    if hospital_id not in HOSPITALS:
        raise PredictionError(
            f"Unknown hospital '{hospital}'. Choose one of: {', '.join(HOSPITALS)}."
        )
    artifact_dir = resolve_variant_dir(output_dir, variant)
    preprocessor = _load_joblib(
        artifact_dir / "shared_preprocessor.joblib", "shared preprocessor artifact"
    )
    pipeline = _load_joblib(
        artifact_dir / f"hospital_{hospital_id}" / "xgboost_pipeline.joblib",
        f"XGBoost model for hospital {hospital_id}",
    )
    return preprocessor, _extract_estimator(pipeline)


def load_hospital_threshold(
    hospital: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    variant: str | None = None,
) -> float:
    """Read the validation-selected decision threshold for one hospital."""
    artifact_dir = resolve_variant_dir(output_dir, variant)
    metadata_path = artifact_dir / f"hospital_{str(hospital).strip().upper()}" / "model_metadata.json"
    if not metadata_path.is_file():
        return 0.5
    try:
        threshold = float(json.loads(metadata_path.read_text(encoding="utf-8")).get("threshold", 0.5))
    except (OSError, ValueError, TypeError):
        return 0.5
    return threshold if 0.0 <= threshold <= 1.0 else 0.5


# --------------------------------------------------------------------------
# Input collection and validation
# --------------------------------------------------------------------------
def _prompt_hospital() -> str:
    """Ask which hospital branch owns this patient's record and model."""
    choices = "/".join(HOSPITALS)
    while True:
        raw = input(f"Select hospital branch ({choices}): ").strip().upper()
        if not raw:
            print("A hospital branch is required so the record stays hospital-local.")
            continue
        if raw not in HOSPITALS:
            print(f"Invalid hospital '{raw}'. Choose one of: {', '.join(HOSPITALS)}.")
            continue
        return raw


def _prompt_identifier() -> str:
    """Ask for a non-empty patient identifier."""
    while True:
        raw = input("Patient ID: ").strip()
        if raw:
            return raw
        print("Patient ID cannot be empty.")


def _prompt_float(feature: str) -> float:
    """Ask for one finite numeric clinical feature."""
    while True:
        raw = input(f"{feature}: ").strip()
        try:
            value = float(raw)
        except ValueError:
            print(f"Invalid value for {feature}: enter a number, for example 72 or 98.6.")
            continue
        if not np.isfinite(value):
            print(f"Invalid value for {feature}: NaN and infinity are not allowed.")
            continue
        return value


def _prompt_category(feature: str, allowed: Sequence[str]) -> str:
    """Ask for one categorical feature from its training vocabulary."""
    options = ", ".join(allowed)
    while True:
        raw = input(f"{feature} ({options}): ").strip()
        for value in allowed:
            if raw.casefold() == value.casefold():
                return value
        print(f"Invalid value for {feature}: choose one of {options}.")


def collect_patient_features(feature_names: Sequence[str]) -> dict[str, Any]:
    """Collect one patient row, matching the saved training schema."""
    features: dict[str, Any] = {}
    categorical_needed: set[str] = set()
    print("\nEnter patient clinical features.\n")
    for feature in feature_names:
        if "=" in feature:
            base = feature.split("=", 1)[0]
            if base in CATEGORICAL_DEFAULTS:
                categorical_needed.add(base)
            continue
        if feature in CATEGORICAL_DEFAULTS:
            features[feature] = _prompt_category(feature, CATEGORICAL_DEFAULTS[feature])
        else:
            features[feature] = _prompt_float(feature)
    # Collect categorical features that only appeared as OneHot expansions.
    for cat in sorted(categorical_needed):
        if cat not in features:
            features[cat] = _prompt_category(cat, CATEGORICAL_DEFAULTS[cat])
    if not features:
        raise PredictionError("No input features were collected; nothing to predict.")
    return features


def _prompt_confirmed_outcome() -> int | None:
    """Ask for the later-confirmed outcome, allowing an unknown answer."""
    while True:
        raw = input("Confirmed outcome if already known (0/1, blank = unknown): ").strip()
        if not raw:
            return None
        if raw in {"0", "1"}:
            return int(raw)
        print("Invalid outcome: enter 0 (no event), 1 (event), or leave blank.")


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def _positive_probability(estimator: Any, transformed: Any) -> float:
    """Return a validated positive-class probability from one model."""
    if not hasattr(estimator, "predict_proba"):
        raise PredictionError(
            f"Loaded artifact {type(estimator).__name__} has no predict_proba method."
        )
    try:
        probabilities = np.asarray(estimator.predict_proba(transformed), dtype=float)
    except Exception as error:  # noqa: BLE001 - surfaced as a clear message
        raise PredictionError(f"The model could not score this patient: {error}") from error
    if probabilities.ndim != 2 or probabilities.shape[0] != 1:
        raise PredictionError(f"Expected one prediction row, received shape {probabilities.shape}.")
    if probabilities.shape[1] == 2:
        classes = getattr(estimator, "classes_", np.array([0, 1]))
        positive = np.flatnonzero(np.asarray(classes) == 1)
        index = int(positive[0]) if positive.size else 1
        probability = float(probabilities[0, index])
    elif probabilities.shape[1] == 1:
        probability = float(probabilities[0, 0])
    else:
        raise PredictionError(f"Expected binary probabilities, received shape {probabilities.shape}.")
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise PredictionError(f"Model returned an invalid probability: {probability!r}.")
    return probability


def predict_for_patient(
    hospital: str,
    patient_id: str,
    features: Mapping[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    store_dir: Path = DEFAULT_STORE_DIR,
    confirmed_outcome: int | None = None,
    variant: str | None = None,
) -> dict[str, Any]:
    """Save, score, and update one patient entirely within one hospital."""
    hospital_id = str(hospital).strip().upper()
    if hospital_id not in HOSPITALS:
        raise PredictionError(
            f"Unknown hospital '{hospital}'. Choose one of: {', '.join(HOSPITALS)}."
        )
    preprocessor, estimator = load_hospital_model(hospital_id, output_dir, variant)
    feature_names = _extract_feature_names(preprocessor)
    threshold = load_hospital_threshold(hospital_id, output_dir, variant)

    # 1. Store the raw record locally, before any prediction is made.
    try:
        initialize_database(hospital_id, store_dir)
        saved = save_patient(hospital_id, patient_id, features, store_dir)
    except DatabaseError as error:
        raise PredictionError(str(error)) from error

    # 2. Transform with the shared pipeline, then score with this hospital's model.
    try:
        frame = pd.DataFrame([dict(features)], columns=list(features))
        transformed = preprocessor.transform(frame)
    except Exception as error:  # noqa: BLE001 - surfaced as a clear message
        raise PredictionError(f"The saved preprocessor rejected this patient: {error}") from error
    probability = _positive_probability(estimator, transformed)

    # 3. Write the prediction back to the same hospital-local row.
    try:
        updated = update_prediction(
            hospital_id, patient_id, probability, threshold=threshold, store_dir=store_dir
        )
        if confirmed_outcome is not None:
            set_confirmed_outcome(hospital_id, patient_id, confirmed_outcome, store_dir)
    except DatabaseError as error:
        raise PredictionError(str(error)) from error

    return {
        "hospital_id": hospital_id,
        "patient_id": saved["patient_id"],
        "database": str(database_path(hospital_id, store_dir)),
        "prediction": updated["prediction"],
        "predicted_label": updated["predicted_label"],
        "threshold": threshold,
        "created_at": saved["created_at"],
        "predicted_at": updated["predicted_at"],
        "confirmed_outcome": confirmed_outcome,
        "feature_count": len(feature_names),
    }


def _print_result(result: Mapping[str, Any]) -> None:
    """Print the stored prediction in a readable block."""
    label = int(result["predicted_label"])
    print("\n" + "=" * 62)
    print(f"  Hospital {result['hospital_id']} local assessment")
    print("=" * 62)
    print(f"  Patient ID          : {result['patient_id']}")
    print(f"  Stored in           : {result['database']}")
    print(f"  Recorded at         : {result['created_at']}")
    print(f"  Predicted at        : {result['predicted_at']}")
    print(f"  Risk probability    : {float(result['prediction']):.4f}")
    print(f"  Risk percentage     : {float(result['prediction']) * 100:.2f}%")
    print(f"  Decision threshold  : {float(result.get('threshold', 0.5)):.4f}")
    print(f"  {TARGET:<20}: {label} ({'higher risk' if label else 'lower risk'})")
    outcome = result.get("confirmed_outcome")
    print(f"  Confirmed outcome   : {'unknown' if outcome is None else outcome}")
    print("\n  " + DISCLAIMER)


def _parse_variant(arguments: Sequence[str]) -> str:
    """Read the optional --variant flag, defaulting to the active variant."""
    for index, argument in enumerate(arguments):
        if argument == "--variant":
            if index + 1 >= len(arguments) or not arguments[index + 1].strip():
                raise PredictionError("--variant requires a value: engineered or onehot.")
            return arguments[index + 1].strip()
        if argument.startswith("--variant="):
            value = argument.split("=", 1)[1].strip()
            if not value:
                raise PredictionError("--variant requires a value: engineered or onehot.")
            return value
    return DEFAULT_VARIANT


def main(argv: Sequence[str] | None = None) -> int:
    """Run the interactive hospital-local intake and prediction workflow."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    print("FedMedTrust patient intake and risk prediction")
    print("Academic prototype using synthetic data; not a medical diagnosis.\n")
    try:
        variant = _parse_variant(arguments)
        artifact_dir = resolve_variant_dir(DEFAULT_OUTPUT_DIR, variant)
        print(f"Artifact variant: {variant}  ({artifact_dir})")
        hospital = _prompt_hospital()
        patient_id = _prompt_identifier()
        preprocessor, _ = load_hospital_model(hospital, variant=variant)
        feature_names = _extract_feature_names(preprocessor)
        features = collect_patient_features(feature_names)
        confirmed_outcome = _prompt_confirmed_outcome()
        result = predict_for_patient(
            hospital,
            patient_id,
            features,
            confirmed_outcome=confirmed_outcome,
            variant=variant,
        )
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled; no further records were written.")
        return 130
    except PredictionError as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
