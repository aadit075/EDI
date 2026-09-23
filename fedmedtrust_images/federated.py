"""FedAvg aggregation for image model weights.

Mirrors the FederatedLogisticTrainer concept from the tabular side:
hospital weight updates are combined into one global model without ever
sharing raw patient images.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class HospitalUpdate:
    """Model update from one hospital after a local training round."""

    hospital: str
    state_dict: Dict[str, Any]
    n_samples: int        # number of training images used
    val_auc: float        # mean AUC on that hospital's local validation set


def fedavg(updates: List[HospitalUpdate]) -> Dict[str, Any]:
    """Federated averaging: weighted mean of state dicts by training set size.

    Parameters
    ----------
    updates:
        One :class:`HospitalUpdate` per participating hospital.

    Returns
    -------
    Averaged state dict suitable for loading into the global model.
    """
    if not updates:
        raise ValueError("fedavg requires at least one hospital update.")

    total = sum(u.n_samples for u in updates)
    if total == 0:
        raise ValueError("All hospital updates have zero samples.")

    avg_state: Dict[str, torch.Tensor] | None = None

    for update in updates:
        weight = update.n_samples / total
        for key, tensor in update.state_dict.items():
            scaled = tensor.float() * weight
            if avg_state is None:
                avg_state = {key: scaled}
            elif key not in avg_state:
                avg_state[key] = scaled
            else:
                avg_state[key] = avg_state[key] + scaled

    assert avg_state is not None
    return avg_state
