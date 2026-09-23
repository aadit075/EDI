"""Image inference utilities for FedMedTrust chest X-ray predictions.

Loads per-hospital ResNet18 models trained on NIH ChestX-ray14 and runs
inference on an uploaded image, returning per-label probabilities for all
14 findings.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

# Canonical label order (must match training order in config.py)
DISEASE_LABELS = (
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
)

# Image preprocessing — identical to validation transform used during training
_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_model_cache: Dict[str, nn.Module] = {}


def _build_resnet18() -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(DISEASE_LABELS))
    return model


def _strip_orig_mod(state_dict: dict) -> dict:
    """Remove _orig_mod. prefix added by torch.compile (if present)."""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def load_image_model(hospital: str, models_dir: Path) -> nn.Module:
    """Load a per-hospital ResNet18 from disk, with in-process caching.

    Falls back to the global model if the per-hospital file is missing.
    """
    key = str(hospital).upper()
    if key in _model_cache:
        return _model_cache[key]

    per_hospital = Path(models_dir) / f"hospital_{key}_round2.pt"
    global_path  = Path(models_dir) / "global_model_latest.pt"

    chosen = per_hospital if per_hospital.exists() else global_path
    if not chosen.exists():
        raise FileNotFoundError(
            f"No model found for hospital {key}. "
            f"Expected {per_hospital} or {global_path}."
        )

    model = _build_resnet18()
    state = torch.load(chosen, map_location="cpu", weights_only=True)
    state = _strip_orig_mod(state)
    model.load_state_dict(state)
    model.eval()

    _model_cache[key] = model
    return model


def predict_xray(
    model: nn.Module,
    image_bytes: bytes,
) -> Dict[str, float]:
    """Run inference on raw image bytes and return per-label probabilities.

    Returns a dict mapping each of the 14 disease labels to a probability
    in [0, 1].  Multi-label: probabilities are independent per label.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _TRANSFORM(img).unsqueeze(0)  # (1, 3, 224, 224)

    with torch.no_grad():
        logits = model(tensor)          # (1, 14)
        probs  = torch.sigmoid(logits)  # multi-label probabilities

    return {
        label: round(float(probs[0, i]), 4)
        for i, label in enumerate(DISEASE_LABELS)
    }


def predict_xray_from_bytes(
    hospital: str,
    image_bytes: bytes,
    models_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """Convenience wrapper: load model by hospital key and run inference."""
    if models_dir is None:
        models_dir = Path(__file__).resolve().parent.parent / "outputs" / "image_models"
    model = load_image_model(hospital, models_dir)
    return predict_xray(model, image_bytes)
