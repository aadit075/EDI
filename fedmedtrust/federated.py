"""Federated logistic learning and trust-aware aggregation."""
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from .metrics import classification_metrics

@dataclass
class ClientUpdate:
    """Contains model parameters and non-sensitive training metadata."""
    hospital: str
    coef: np.ndarray
    intercept: np.ndarray
    samples: int
    validation_score: float


def adaptive_trust_aggregate(updates: Sequence[ClientUpdate], previous: Tuple[np.ndarray, np.ndarray] | None = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Aggregate updates using sample weight and distance-based adaptive trust."""
    if not updates:
        raise ValueError("At least one client update is required")
    reference = previous[0] if previous is not None else updates[0].coef
    distances = np.asarray([float(np.linalg.norm(item.coef - reference)) for item in updates])
    trust = 1.0 / (1.0 + distances)
    weights = trust * np.asarray([max(0, item.samples) for item in updates], dtype=float)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = trust
    weights /= weights.sum()
    coef = sum(item.coef * weight for item, weight in zip(updates, weights))
    intercept = sum(item.intercept * weight for item, weight in zip(updates, weights))
    return coef, intercept, {item.hospital: float(weight) for item, weight in zip(updates, weights)}

class FederatedLogisticTrainer:
    """Runs FedAvg or adaptive-trust logistic regression without central records."""
    def __init__(self, rounds: int = 5, max_iter: int = 200, seed: int = 42) -> None:
        self.rounds = max(1, rounds); self.max_iter = max(20, max_iter); self.seed = seed
        self.coef_: np.ndarray | None = None; self.intercept_: np.ndarray | None = None
        self.history_: List[Dict[str, float]] = []

    def fit(self, clients: Mapping[str, Tuple[np.ndarray, np.ndarray]], trust: bool = True) -> "FederatedLogisticTrainer":
        """Train from per-hospital matrices; client rows never leave their hospital."""
        valid = {name: (x, np.asarray(y, dtype=int)) for name, (x, y) in clients.items() if len(x) and np.unique(y).size >= 2}
        if not valid: raise ValueError("No client has non-empty binary training data")
        n_features = next(iter(valid.values()))[0].shape[1]
        self.coef_ = np.zeros((1, n_features), dtype=float); self.intercept_ = np.zeros(1, dtype=float)
        for round_number in range(1, self.rounds + 1):
            updates: List[ClientUpdate] = []
            for hospital, (x, y) in valid.items():
                model = LogisticRegression(max_iter=self.max_iter, class_weight="balanced", random_state=self.seed)
                model.fit(x, y)
                score = float(classification_metrics(y, model.predict_proba(x)[:, 1])["f1"])
                updates.append(ClientUpdate(hospital, model.coef_.copy(), model.intercept_.copy(), len(y), score))
            if trust:
                self.coef_, self.intercept_, weights = adaptive_trust_aggregate(updates, (self.coef_, self.intercept_))
            else:
                total = sum(item.samples for item in updates); weights = {item.hospital: item.samples / total for item in updates}
                self.coef_ = sum(item.coef * weights[item.hospital] for item in updates); self.intercept_ = sum(item.intercept * weights[item.hospital] for item in updates)
            self.history_.append({"round": float(round_number), "mean_local_f1": float(np.mean([u.validation_score for u in updates])), "mean_trust": float(np.mean(list(weights.values())))})
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict positive-class probabilities using the aggregated model."""
        if self.coef_ is None or self.intercept_ is None: raise RuntimeError("Trainer is not fitted")
        scores = np.asarray(features) @ self.coef_[0] + self.intercept_[0]
        scores = np.clip(scores, -700, 700)
        positive = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack((1.0 - positive, positive))
