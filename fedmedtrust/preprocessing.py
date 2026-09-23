"""Schema-driven preprocessing with no centralized patient pooling."""
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class FeatureMetadata:
    """Defines a shared feature schema and safe domain defaults."""
    numeric: Sequence[str]
    categorical: Mapping[str, Sequence[str]]
    excluded: Sequence[str]

class SharedPreprocessor:
    """Transforms hospital frames into one aligned numeric feature space.

    The schema and category vocabularies are supplied as metadata; no patient
    rows are combined and no statistics are fitted on a central dataset.
    """
    def __init__(self, metadata: FeatureMetadata) -> None:
        self.metadata = metadata
        self.feature_names_: List[str] = list(metadata.numeric)
        for column, values in metadata.categorical.items():
            self.feature_names_.extend(f"{column}={value}" for value in values)
        self._numeric_scales: Dict[str, float] = {column: 1.0 for column in metadata.numeric}

    @classmethod
    def from_schema(cls, columns: Iterable[str], excluded: Optional[Sequence[str]] = None) -> "SharedPreprocessor":
        """Build metadata from column names and fixed categorical vocabularies."""
        excluded_set = set(excluded or ("patient_id", "hospital_id", "cardiac_risk_label"))
        categorical_values = {"sex": ("Female", "Male", "Other", "Unknown"), "smoking_status": ("never", "former", "current", "Unknown"), "physical_activity_level": ("low", "moderate", "high", "Unknown")}
        numeric: List[str] = []
        categorical: Dict[str, Sequence[str]] = {}
        for column in columns:
            if column in excluded_set:
                continue
            if column in categorical_values:
                categorical[column] = categorical_values[column]
            else:
                numeric.append(column)
        return cls(FeatureMetadata(numeric, categorical, tuple(excluded_set)))

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Transform one hospital's records without sharing or pooling rows."""
        blocks: List[np.ndarray] = []
        for column in self.metadata.numeric:
            series = frame[column] if column in frame.columns else pd.Series(0.0, index=frame.index)
            values = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
            scale = self._numeric_scales.get(column, 1.0)
            blocks.append((values / scale).reshape(-1, 1))
        for column, categories in self.metadata.categorical.items():
            series = frame[column] if column in frame.columns else pd.Series("Unknown", index=frame.index)
            values = series.fillna("Unknown").astype(str).to_numpy()
            blocks.append(np.column_stack([(values == category).astype(float) for category in categories]))
        return np.hstack(blocks) if blocks else np.empty((len(frame), 0), dtype=float)

    def feature_names(self) -> List[str]:
        """Return the stable global feature ordering."""
        return list(self.feature_names_)
