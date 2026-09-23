"""Shared feature-space preprocessing for FedMedTrust training and inference.

Two spaces are supported so the same object type is used at training and
prediction time:
  * EngineeredPreprocessor - numeric columns plus derived clinical ratios and
    interactions, with categorical fields expanded into indicators.
  * SharedPreprocessor     - the schema-driven space in preprocessing.py.

Living in the package, rather than in a script, keeps joblib pickles loadable
from any entry point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd

CATEGORICAL_VALUES: Dict[str, Sequence[str]] = {
    "sex": ("Female", "Male", "Other", "Unknown"),
    "smoking_status": ("never", "former", "current", "Unknown"),
    "physical_activity_level": ("low", "moderate", "high", "Unknown"),
}

ENGINEERED_NUMERIC: tuple[str, ...] = (
    "pulse_pressure",
    "mean_arterial_pressure",
    "cholesterol_hdl_ratio",
    "ldl_hdl_ratio",
    "age_bmi_interaction",
    "age_squared",
    "bmi_squared",
    "glucose_hba1c_interaction",
)


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the clinical ratios and interactions used by the exported models."""
    df = frame.copy()
    for column in (
        "height_cm", "weight_kg", "age", "bmi", "systolic_bp", "diastolic_bp",
        "total_cholesterol", "hdl_cholesterol", "ldl_cholesterol",
        "blood_glucose", "hba1c",
    ):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    if {"systolic_bp", "diastolic_bp"} <= set(df.columns):
        df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
        df["mean_arterial_pressure"] = df["diastolic_bp"] + (
            df["systolic_bp"] - df["diastolic_bp"]
        ) / 3.0
    if {"total_cholesterol", "hdl_cholesterol"} <= set(df.columns):
        df["cholesterol_hdl_ratio"] = df["total_cholesterol"] / df["hdl_cholesterol"].replace(0, np.nan)
    if {"ldl_cholesterol", "hdl_cholesterol"} <= set(df.columns):
        df["ldl_hdl_ratio"] = df["ldl_cholesterol"] / df["hdl_cholesterol"].replace(0, np.nan)
    if {"age", "bmi"} <= set(df.columns):
        df["age_bmi_interaction"] = df["age"] * df["bmi"]
    if "age" in df.columns:
        df["age_squared"] = df["age"] ** 2
    if "bmi" in df.columns:
        df["bmi_squared"] = df["bmi"] ** 2
    if {"blood_glucose", "hba1c"} <= set(df.columns):
        df["glucose_hba1c_interaction"] = df["blood_glucose"] * df["hba1c"]
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


@dataclass
class EngineeredPreprocessor:
    """Numeric plus engineered features, with indicator-expanded categoricals."""

    numeric: tuple[str, ...]
    categorical: Mapping[str, Sequence[str]] = field(default_factory=lambda: dict(CATEGORICAL_VALUES))
    engineered: tuple[str, ...] = ENGINEERED_NUMERIC

    def feature_names(self) -> list[str]:
        """Return the stable feature ordering used by the trained models."""
        names = list(self.numeric)
        for column, values in self.categorical.items():
            names.extend(f"{column}={value}" for value in values)
        return names

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Build one aligned numeric matrix from a raw patient frame."""
        df = add_engineered_features(frame)
        blocks = []
        for column in self.numeric:
            if column not in df.columns:
                raise KeyError(f"missing expected feature column '{column}'")
            blocks.append(pd.to_numeric(df[column], errors="coerce").fillna(0.0).to_numpy(dtype=float).reshape(-1, 1))
        for column, categories in self.categorical.items():
            if column not in df.columns:
                raise KeyError(f"missing expected categorical column '{column}'")
            values = df[column].fillna("Unknown").astype(str).to_numpy()
            blocks.append(np.column_stack([(values == category).astype(float) for category in categories]))
        return np.hstack(blocks) if blocks else np.empty((len(df), 0), dtype=float)
