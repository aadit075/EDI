"""Configuration for the FedMedTrust image extension."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

# NIH ChestX-ray14 disease labels (same order as the dataset paper).
DISEASE_LABELS: Tuple[str, ...] = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
)


@dataclass(frozen=True)
class ImageProjectConfig:
    """Paths and hyperparameters for one federated image training run."""

    hospitals: Tuple[str, ...] = ("A", "B", "C", "D", "E")
    labels: Tuple[str, ...] = DISEASE_LABELS

    # Federated settings
    rounds: int = 3
    epochs_per_round: int = 3

    # Training settings
    batch_size: int = 64
    learning_rate: float = 1e-4
    random_seed: int = 42

    # Modal volume mount point
    volume_path: str = "/data"

    # ── Derived paths (all inside the Modal volume) ──────────────────────────

    @property
    def images_dir(self) -> Path:
        return Path(self.volume_path) / "images"

    @property
    def metadata_path(self) -> Path:
        return Path(self.volume_path) / "Data_Entry_2017.csv"

    @property
    def splits_path(self) -> Path:
        return Path(self.volume_path) / "hospital_splits.csv"

    @property
    def models_dir(self) -> Path:
        return Path(self.volume_path) / "models"

    @property
    def results_dir(self) -> Path:
        return Path(self.volume_path) / "results"

    @property
    def global_model_path(self) -> Path:
        return self.models_dir / "global_model_latest.pt"
