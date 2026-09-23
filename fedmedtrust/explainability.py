"""Optional explainability and XGBoost strategy."""
from typing import Any, Dict, Optional
import numpy as np

def explain_prediction(model: Any, features: np.ndarray, feature_names: list[str]) -> Dict[str, float]:
    """Return SHAP values when installed, otherwise coefficient contributions."""
    try:
        import shap  # type: ignore
        values = shap.Explainer(model, features)(features).values
        values = np.asarray(values); values = values[:, :, 1] if values.ndim == 3 else values
        return {name: float(np.mean(np.abs(values[:, index]))) for index, name in enumerate(feature_names)}
    except (ImportError, RuntimeError, AttributeError, ValueError):
        coefficients = getattr(model, "coef_", np.zeros((1, len(feature_names))))[0]
        return {name: float(abs(coefficients[index])) for index, name in enumerate(feature_names)}

class FederatedXGBoostStrategy:
    """Optional round-wise local booster ensemble; never averages tree coefficients."""
    def __init__(self, rounds: int = 3, seed: int = 42) -> None:
        self.rounds = max(1, rounds); self.seed = seed; self.models: list[Any] = []

    def fit(self, clients: Dict[str, tuple[np.ndarray, np.ndarray]]) -> "FederatedXGBoostStrategy":
        """Fit local XGBoost boosters and retain them as a prediction ensemble."""
        try:
            from xgboost import XGBClassifier  # type: ignore
        except ImportError:
            self.models = []
            return self
        self.models = []
        for _, (features, labels) in clients.items():
            if len(features) and np.unique(labels).size > 1:
                model = XGBClassifier(n_estimators=30, max_depth=3, learning_rate=0.08, n_jobs=1, tree_method="hist", random_state=self.seed, eval_metric="logloss")
                model.fit(features, labels); self.models.append(model)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Average predictions from retained local boosters, or return neutral scores."""
        if not self.models: return np.tile(np.array([[0.5, 0.5]]), (len(features), 1))
        positive = np.mean([model.predict_proba(features)[:, 1] for model in self.models], axis=0)
        return np.column_stack((1.0 - positive, positive))
