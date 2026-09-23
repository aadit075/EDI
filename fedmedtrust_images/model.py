"""ResNet18 model definition and weight serialisation helpers."""
from __future__ import annotations

import io
from typing import Any, Dict

import torch
import torch.nn as nn


def build_model(num_classes: int = 14, pretrained: bool = True) -> nn.Module:
    """Return ResNet18 with a linear head replaced for ``num_classes`` outputs.

    The model is suitable for multi-label classification: use
    ``BCEWithLogitsLoss`` and ``sigmoid`` at inference time.
    """
    import torchvision.models as models

    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def weights_to_bytes(state_dict: Dict[str, Any]) -> bytes:
    """Serialise a model state dict to raw bytes for transport between Modal functions."""
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return buf.getvalue()


def bytes_to_weights(data: bytes) -> Dict[str, Any]:
    """Deserialise a state dict previously produced by :func:`weights_to_bytes`."""
    return torch.load(io.BytesIO(data), map_location="cpu")
