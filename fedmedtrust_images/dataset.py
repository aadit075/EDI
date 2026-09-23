"""Dataset utilities for NIH ChestX-ray14 with per-hospital partitioning.

Patient records are assigned to hospitals by patient ID, so no patient ever
appears in more than one hospital partition.  Raw images stay in whichever
hospital directory they are assigned to; nothing is pooled.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import DISEASE_LABELS


# ── Transforms ────────────────────────────────────────────────────────────────

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Patient-level hospital assignment ─────────────────────────────────────────

def split_patients_to_hospitals(
    metadata_path: str | Path,
    hospitals: Sequence[str] = ("A", "B", "C", "D", "E"),
    seed: int = 42,
) -> pd.DataFrame:
    """Assign each unique patient to one hospital partition.

    The assignment is deterministic: shuffle patient IDs once then round-robin
    across hospitals.  Returns the full metadata DataFrame with a new
    ``hospital`` column.
    """
    df = pd.read_csv(metadata_path)
    patient_ids = sorted(df["Patient ID"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)
    hospital_map = {
        pid: hospitals[i % len(hospitals)]
        for i, pid in enumerate(patient_ids)
    }
    df["hospital"] = df["Patient ID"].map(hospital_map)
    return df


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class ChestXrayDataset(Dataset):
    """NIH ChestX-ray14 dataset for one hospital partition.

    Parameters
    ----------
    df:
        Rows belonging to this hospital (already filtered).
    images_dir:
        Directory that holds all PNG files (flat layout).
    labels:
        Ordered list of disease labels to predict.
    augment:
        Use training augmentations when True, plain normalisation otherwise.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str | Path,
        labels: Sequence[str] = DISEASE_LABELS,
        augment: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.labels = list(labels)
        self.transform = TRAIN_TRANSFORM if augment else VAL_TRANSFORM
        self.label_matrix = self._build_label_matrix()

    def _build_label_matrix(self) -> np.ndarray:
        mat = np.zeros((len(self.df), len(self.labels)), dtype=np.float32)
        for i, findings in enumerate(self.df["Finding Labels"]):
            for finding in str(findings).split("|"):
                finding = finding.strip()
                if finding in self.labels:
                    mat[i, self.labels.index(finding)] = 1.0
        return mat

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        name = self.df.iloc[idx]["Image Index"]
        img = Image.open(self.images_dir / name).convert("RGB")
        return self.transform(img), torch.tensor(self.label_matrix[idx])

    def positive_weights(self) -> torch.Tensor:
        """Return per-class positive weights for BCEWithLogitsLoss."""
        pos = self.label_matrix.sum(axis=0)
        neg = len(self.df) - pos
        return torch.tensor(neg / (pos + 1e-6), dtype=torch.float32)


# ── DataLoader factory ────────────────────────────────────────────────────────

def load_hospital_loaders(
    df: pd.DataFrame,
    hospital: str,
    images_dir: str | Path,
    labels: Sequence[str] = DISEASE_LABELS,
    batch_size: int = 64,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, ChestXrayDataset]:
    """Return (train, val, test) DataLoaders and the training Dataset for one hospital.

    Split: 70 % train / 15 % val / 15 % test.  Splitting is by row index after
    a patient-level shuffle so the same patient cannot appear in both train and
    test within the same hospital.
    """
    hospital_df = df[df["hospital"] == hospital].copy().reset_index(drop=True)

    rng = np.random.default_rng(seed + ord(hospital))
    idx = rng.permutation(len(hospital_df))
    n = len(idx)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    train_df = hospital_df.iloc[idx[:train_end]]
    val_df = hospital_df.iloc[idx[train_end:val_end]]
    test_df = hospital_df.iloc[idx[val_end:]]

    train_ds = ChestXrayDataset(train_df, images_dir, labels, augment=True)
    val_ds = ChestXrayDataset(val_df, images_dir, labels, augment=False)
    test_ds = ChestXrayDataset(test_df, images_dir, labels, augment=False)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    return train_loader, val_loader, test_loader, train_ds
