"""Multi-label classification metrics for chest X-ray predictions."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: Sequence[str],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute per-label and mean AUC / F1 for multi-label classification.

    Parameters
    ----------
    y_true:
        Binary ground-truth matrix of shape (n_samples, n_labels).
    y_prob:
        Predicted probability matrix of shape (n_samples, n_labels).
    labels:
        Ordered label names matching the columns of the matrices.
    threshold:
        Decision threshold for F1 computation.

    Returns
    -------
    Dictionary with ``<label>_auc``, ``<label>_f1``, ``mean_auc``,
    ``mean_f1`` entries.  Labels with no positive or no negative examples
    are skipped (NaN).
    """
    from sklearn.metrics import f1_score, roc_auc_score

    results: Dict[str, float] = {}
    aucs: List[float] = []
    f1s: List[float] = []

    for i, label in enumerate(labels):
        t = y_true[:, i]
        p = y_prob[:, i]

        # Skip degenerate columns
        if t.sum() == 0 or t.sum() == len(t):
            results[f"{label}_auc"] = float("nan")
            results[f"{label}_f1"] = float("nan")
            continue

        auc = float(roc_auc_score(t, p))
        f1 = float(f1_score(t, (p >= threshold).astype(int), zero_division=0))

        results[f"{label}_auc"] = auc
        results[f"{label}_f1"] = f1
        aucs.append(auc)
        f1s.append(f1)

    results["mean_auc"] = float(np.nanmean(aucs)) if aucs else float("nan")
    results["mean_f1"] = float(np.nanmean(f1s)) if f1s else float("nan")
    return results
