"""Configuration for FedMedTrust."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

@dataclass(frozen=True)
class ProjectConfig:
    """Holds paths and model settings for a local federated run."""
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    target: str = "cardiac_risk_label"
    hospitals: Tuple[str, ...] = ("A", "B", "C", "D", "E")
    rounds: int = 5
    random_seed: int = 42
    max_iter: int = 200
    @property
    def data_dir(self) -> Path:
        """Return the dataset directory."""
        return self.root / "Data"

    @property
    def model_artifact(self) -> Path:
        """Return the persisted federated model artifact path."""
        return self.root / "federated_model.joblib"

    @property
    def history_artifact(self) -> Path:
        """Return the persisted training history path."""
        return self.root / "federated_training_history.csv"
