"""Retrain the five hospital XGBoost models and persist a shared preprocessor.

Two feature spaces are trained and compared:
  * engineered: 8 derived clinical ratios/interactions plus the 3 categorical
    fields kept as raw strings, matching the contract recorded in the
    existing outputs_xgboost_target80 model metadata (42 columns).
  * onehot: the repo's SharedPreprocessor space, where the 3 categorical
    fields expand into 12 indicator columns with no feature engineering.

Nothing is written to outputs/ until a feature space is selected, so the
comparison is safe to run repeatedly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from fedmedtrust.feature_spaces import ENGINEERED_NUMERIC, EngineeredPreprocessor  # noqa: E402
DATA_DIR = ROOT / "Data"
OUTPUT_DIR = ROOT / "outputs"

TARGET = "cardiac_risk_label"
HOSPITALS: tuple[str, ...] = ("A", "B", "C", "D", "E")
EXCLUDED = {"patient_id", "hospital_id", TARGET}
CATEGORICALS: dict[str, tuple[str, ...]] = {
    "sex": ("Female", "Male", "Other", "Unknown"),
    "smoking_status": ("never", "former", "current", "Unknown"),
    "physical_activity_level": ("low", "moderate", "high", "Unknown"),
}
SEED = 42
PARAMS: dict[str, Any] = {
    "n_estimators": 150,
    "max_depth": 2,
    "learning_rate": 0.05,
    "min_child_weight": 2,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 5.0,
    "reg_alpha": 0.0,
    "random_state": SEED,
    "n_jobs": 4,
    "tree_method": "hist",
    "eval_metric": "logloss",
}


class TrainingError(RuntimeError):
    """Raised when training data or configuration is unusable."""


# --------------------------------------------------------------------------
# Feature space selection
# --------------------------------------------------------------------------
def build_preprocessor(space: str, columns: Sequence[str]) -> Any:
    """Build the preprocessor for the requested feature space."""
    if space == "engineered":
        numeric = tuple(c for c in columns if c not in EXCLUDED and c not in CATEGORICALS)
        return EngineeredPreprocessor(numeric=numeric, categorical=dict(CATEGORICALS))
    if space == "onehot":
        sys.path.insert(0, str(ROOT))
        from fedmedtrust.preprocessing import SharedPreprocessor

        return SharedPreprocessor.from_schema(columns)
    raise TrainingError(f"Unknown feature space '{space}'. Use 'engineered' or 'onehot'.")


# --------------------------------------------------------------------------
# Training and evaluation
# --------------------------------------------------------------------------
def _auc(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Return ROC AUC and PR AUC, tolerating degenerate labels."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    if np.unique(y_true).size < 2:
        return 0.5, 0.0
    return float(roc_auc_score(y_true, probabilities)), float(average_precision_score(y_true, probabilities))


def _metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute the standard binary classification metrics."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    predicted = (probabilities >= threshold).astype(int)
    roc_auc, pr_auc = _auc(y_true, probabilities)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def _load(hospital: str, split: str) -> pd.DataFrame:
    """Load one hospital split from Data/."""
    path = DATA_DIR / f"hospital_{hospital}_{split}.csv"
    if not path.is_file():
        raise TrainingError(f"Missing dataset: {path}")
    return pd.read_csv(path)


def train_space(space: str) -> tuple[list[dict[str, Any]], dict[str, tuple[Any, Any]]]:
    """Train all five hospitals in one feature space and return metrics."""
    reference = pd.read_csv(DATA_DIR / "hospital_A_train.csv", nrows=0).columns
    preprocessor = build_preprocessor(space, reference)
    feature_names = preprocessor.feature_names()
    results: list[dict[str, Any]] = []
    fitted: dict[str, tuple[Any, Any]] = {}

    for hospital in HOSPITALS:
        train = _load(hospital, "train")
        validation = _load(hospital, "validation")
        test = _load(hospital, "test")
        x_train = preprocessor.transform(train)
        y_train = train[TARGET].astype(int).to_numpy()
        x_val = preprocessor.transform(validation)
        y_val = validation[TARGET].astype(int).to_numpy()
        x_test = preprocessor.transform(test)
        y_test = test[TARGET].astype(int).to_numpy()

        model = XGBClassifier(**PARAMS)
        model.fit(x_train, y_train)

        # Select the validation threshold that maximizes validation accuracy.
        validation_probabilities = model.predict_proba(x_val)[:, 1]
        candidates = np.unique(np.round(validation_probabilities, 6))
        best_threshold, best_score = 0.5, -1.0
        for candidate in candidates:
            score = float(((validation_probabilities >= candidate).astype(int) == y_val).mean())
            if score > best_score:
                best_threshold, best_score = float(candidate), score

        test_probabilities = model.predict_proba(x_test)[:, 1]
        result = {"hospital": hospital, "threshold": best_threshold,
                  "train_rows": len(y_train), "features": len(feature_names)}
        result.update({f"test_{k}": v for k, v in _metrics(y_test, test_probabilities, best_threshold).items()})
        results.append(result)
        fitted[hospital] = (model, best_threshold)
        print(
            f"    hospital {hospital}: acc={result['test_accuracy']:.4f} "
            f"f1={result['test_f1']:.4f} roc_auc={result['test_roc_auc']:.4f} "
            f"(threshold={best_threshold:.4f})"
        )
    return results, fitted


def compare() -> dict[str, list[dict[str, Any]]]:
    """Train both feature spaces and print a side-by-side comparison."""
    everything: dict[str, list[dict[str, Any]]] = {}
    for space in ("engineered", "onehot"):
        print(f"\nTraining feature space: {space}")
        results, _ = train_space(space)
        everything[space] = results
        accuracy = np.mean([r["test_accuracy"] for r in results])
        f1 = np.mean([r["test_f1"] for r in results])
        print(f"  mean accuracy={accuracy:.4f}  mean f1={f1:.4f}")

    print("\n" + "=" * 74)
    print("  ACCURACY COMPARISON (test set)")
    print("=" * 74)
    print(f"  {'Hospital':<10}{'engineered':>14}{'onehot':>14}{'difference':>14}")
    print("-" * 74)
    for index, hospital in enumerate(HOSPITALS):
        eng = everything["engineered"][index]["test_accuracy"]
        one = everything["onehot"][index]["test_accuracy"]
        print(f"  {hospital:<10}{eng:>14.4f}{one:>14.4f}{eng - one:>+14.4f}")
    eng_mean = np.mean([r["test_accuracy"] for r in everything["engineered"]])
    one_mean = np.mean([r["test_accuracy"] for r in everything["onehot"]])
    print("-" * 74)
    print(f"  {'MEAN':<10}{eng_mean:>14.4f}{one_mean:>14.4f}{eng_mean - one_mean:>+14.4f}")
    print(f"\n  engineered features: {everything['engineered'][0]['features']}")
    print(f"  onehot features    : {everything['onehot'][0]['features']}")
    return everything


def save(space: str, output_dir: Path | None = None) -> None:
    """Train the chosen space and write its artifacts to an output directory."""
    destination = Path(output_dir) if output_dir else OUTPUT_DIR
    print(f"\nSaving feature space '{space}' -> {destination}")
    results, fitted = train_space(space)
    reference = pd.read_csv(DATA_DIR / "hospital_A_train.csv", nrows=0).columns
    preprocessor = build_preprocessor(space, reference)

    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, destination / "shared_preprocessor.joblib")
    summary: list[dict[str, Any]] = []
    for hospital in HOSPITALS:
        model, threshold = fitted[hospital]
        directory = destination / f"hospital_{hospital}"
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, directory / "xgboost_pipeline.joblib")
        metadata = {
            "hospital_id": hospital,
            "model": "xgboost",
            "feature_space": space,
            "features": preprocessor.feature_names(),
            "threshold": threshold,
            "training_rows": int((pd.read_csv(DATA_DIR / f"hospital_{hospital}_train.csv",
                                              usecols=[TARGET]).shape[0])),
            "parameters": PARAMS,
            "metrics": {k: v for k, v in results[HOSPITALS.index(hospital)].items()
                        if k.startswith("test_")},
        }
        (directory / "model_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        summary.append({"hospital": hospital, **metadata["metrics"]})
    pd.DataFrame(summary).to_csv(destination / "final_test_summary.csv", index=False)
    print(f"[OK] shared preprocessor -> {destination / 'shared_preprocessor.joblib'}")
    print(f"[OK] five models         -> {destination / 'hospital_<A-E>'}")
    print(f"[OK] summary             -> {destination / 'final_test_summary.csv'}")


def save_all() -> None:
    """Train and save both feature spaces into separate, comparable folders."""
    targets = {
        "engineered": OUTPUT_DIR / "engineered",
        "onehot": OUTPUT_DIR / "onehot",
    }
    for space, destination in targets.items():
        save(space, destination)
    combined = pd.read_csv(OUTPUT_DIR / "onehot" / "final_test_summary.csv")
    comparison = pd.read_csv(OUTPUT_DIR / "engineered" / "final_test_summary.csv")
    print("\n" + "=" * 74)
    print("  SAVED ARTIFACT COMPARISON (test accuracy)")
    print("=" * 74)
    print(f"  {'Hospital':<10}{'engineered':>14}{'onehot':>14}{'difference':>14}")
    print("-" * 74)
    for hospital in HOSPITALS:
        eng = float(comparison.loc[comparison["hospital"] == hospital, "test_accuracy"].iloc[0])
        one = float(combined.loc[combined["hospital"] == hospital, "test_accuracy"].iloc[0])
        print(f"  {hospital:<10}{eng:>14.4f}{one:>14.4f}{eng - one:>+14.4f}")
    print("-" * 74)
    print(f"  {'MEAN':<10}{comparison['test_accuracy'].mean():>14.4f}"
          f"{combined['test_accuracy'].mean():>14.4f}"
          f"{comparison['test_accuracy'].mean() - combined['test_accuracy'].mean():>+14.4f}")
    print(f"\n  Selection at prediction time: python predict_patient.py --variant engineered")


def main(argv: Sequence[str] | None = None) -> int:
    """Compare feature spaces, or save one or both of them."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not DATA_DIR.is_dir():
        print(f"ERROR: dataset directory not found: {DATA_DIR}", file=sys.stderr)
        return 1
    try:
        if arguments and arguments[0] == "--save-all":
            save_all()
            return 0
        if arguments and arguments[0] in {"--save", "--save-engineered", "--save-onehot"}:
            mapping = {
                "--save-engineered": ("engineered", OUTPUT_DIR / "engineered"),
                "--save-onehot": ("onehot", OUTPUT_DIR / "onehot"),
                "--save": (
                    arguments[1] if len(arguments) > 1 else "engineered",
                    OUTPUT_DIR / (arguments[1] if len(arguments) > 1 else "engineered"),
                ),
            }
            save(*mapping[arguments[0]])
            return 0
        compare()
    except TrainingError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("\nNothing saved. Choose a space, then run:")
    print("  python retrain_hospitals.py --save-all")
    print("  python retrain_hospitals.py --save engineered")
    print("  python retrain_hospitals.py --save onehot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
