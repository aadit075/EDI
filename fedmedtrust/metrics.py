"""Metrics for local and federated evaluation."""
from typing import Any, Dict
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

def classification_metrics(y_true: Any, probabilities: Any, threshold: float = 0.5) -> Dict[str, float]:
    """Compute robust binary classification metrics, including degenerate labels."""
    y = np.asarray(y_true, dtype=int); p = np.asarray(probabilities, dtype=float)
    pred = (p >= threshold).astype(int)
    result = {"accuracy": float(accuracy_score(y, pred)), "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)), "f1": float(f1_score(y, pred, zero_division=0))}
    result["roc_auc"] = float(roc_auc_score(y, p)) if np.unique(y).size > 1 else 0.5
    result["pr_auc"] = float(average_precision_score(y, p)) if y.size else 0.0
    return result
