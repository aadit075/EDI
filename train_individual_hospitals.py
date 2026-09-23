"""Train separate logistic regression models for each individual hospital.

Each hospital gets its own independent LogisticRegression model trained on
its local training data, evaluated on its test and validation sets, and
saved as a standalone artifact.
"""
from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

from fedmedtrust.config import ProjectConfig
from fedmedtrust.preprocessing import SharedPreprocessor


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Compute classification metrics for binary prediction."""
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if np.unique(y_true).size > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = 0.5
        metrics["pr_auc"] = 0.0
    return metrics


def main() -> None:
    config = ProjectConfig()
    data_dir = config.data_dir
    target = config.target
    output_dir = config.root / "individual_hospital_models"
    os.makedirs(output_dir, exist_ok=True)

    # Build preprocessor from schema (using Hospital A columns as reference)
    ref_columns = pd.read_csv(data_dir / "hospital_A_train.csv", nrows=0).columns
    preprocessor = SharedPreprocessor.from_schema(ref_columns)

    all_results = []

    print("=" * 70)
    print("  INDIVIDUAL HOSPITAL LOGISTIC REGRESSION TRAINING")
    print("=" * 70)

    for hospital in config.hospitals:
        print(f"\n{'-' * 70}")
        print(f"  Hospital {hospital}")
        print(f"{'-' * 70}")

        # -- Load data ------------------------------------------------------
        train_df = pd.read_csv(data_dir / f"hospital_{hospital}_train.csv")
        val_df = pd.read_csv(data_dir / f"hospital_{hospital}_validation.csv")
        test_df = pd.read_csv(data_dir / f"hospital_{hospital}_test.csv")

        X_train = preprocessor.transform(train_df)
        y_train = train_df[target].astype(int).to_numpy()

        X_val = preprocessor.transform(val_df)
        y_val = val_df[target].astype(int).to_numpy()

        X_test = preprocessor.transform(test_df)
        y_test = test_df[target].astype(int).to_numpy()

        print(f"  Train samples : {len(y_train)}  (pos={y_train.sum()}, neg={len(y_train) - y_train.sum()})")
        print(f"  Val   samples : {len(y_val)}")
        print(f"  Test  samples : {len(y_test)}")

        # -- Train logistic regression --------------------------------------
        model = LogisticRegression(
            max_iter=config.max_iter,
            class_weight="balanced",
            random_state=config.random_seed,
            solver="lbfgs",
        )
        model.fit(X_train, y_train)

        # -- Evaluate on all splits -----------------------------------------
        train_prob = model.predict_proba(X_train)[:, 1]
        val_prob = model.predict_proba(X_val)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]

        train_metrics = evaluate(y_train, train_prob)
        val_metrics = evaluate(y_val, val_prob)
        test_metrics = evaluate(y_test, test_prob)

        print(f"\n  +-- Train Metrics -----------------------------------------------")
        for k, v in train_metrics.items():
            print(f"  |  {k:>12s}: {v:.4f}")

        print(f"  +-- Validation Metrics ------------------------------------------")
        for k, v in val_metrics.items():
            print(f"  |  {k:>12s}: {v:.4f}")

        print(f"  +-- Test Metrics ------------------------------------------------")
        for k, v in test_metrics.items():
            print(f"  |  {k:>12s}: {v:.4f}")

        # -- Classification report on test set ------------------------------
        y_test_pred = (test_prob >= 0.5).astype(int)
        print(f"  |")
        print(f"  +-- Classification Report (Test) --------------------------------")
        report = classification_report(y_test, y_test_pred, target_names=["No Risk", "Risk"])
        for line in report.split("\n"):
            print(f"  |  {line}")

        print(f"  +-- Confusion Matrix (Test) -------------------------------------")
        cm = confusion_matrix(y_test, y_test_pred)
        print(f"  |                  Predicted")
        print(f"  |                 No Risk  Risk")
        print(f"  |  Actual No Risk   {cm[0][0]:>5d}  {cm[0][1]:>5d}")
        print(f"  |  Actual Risk      {cm[1][0]:>5d}  {cm[1][1]:>5d}")
        print(f"  +---------------------------------------------------------------")

        # ── Save per-hospital model ────────────────────────────────────────
        model_path = output_dir / f"hospital_{hospital}_model.joblib"
        joblib.dump(
            {
                "hospital": hospital,
                "model": model,
                "preprocessor": preprocessor,
                "feature_names": preprocessor.feature_names(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "trained_at": pd.Timestamp.utcnow().isoformat(),
            },
            model_path,
        )
        print(f"\n  [OK] Model saved -> {model_path}")

        # ── Collect for summary ────────────────────────────────────────────
        result = {"hospital": hospital}
        for split, m in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
            for k, v in m.items():
                result[f"{split}_{k}"] = v
        all_results.append(result)

    # ── Save summary CSV ───────────────────────────────────────────────────
    summary_df = pd.DataFrame(all_results)
    summary_path = output_dir / "individual_hospital_results.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\n\n{'=' * 70}")
    print("  SUMMARY - Test Set Performance Across Hospitals")
    print(f"{'=' * 70}")
    test_cols = ["hospital"] + [c for c in summary_df.columns if c.startswith("test_")]
    print(summary_df[test_cols].to_string(index=False, float_format="{:.4f}".format))
    print(f"\n  [OK] All results saved -> {summary_path}")
    print(f"  [OK] Individual models  -> {output_dir}")


if __name__ == "__main__":
    main()
