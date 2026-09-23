"""Train and persist the FedMedTrust federated model."""
from __future__ import annotations

import joblib
import pandas as pd
from fedmedtrust.config import ProjectConfig
from fedmedtrust.preprocessing import SharedPreprocessor
from fedmedtrust.federated import FederatedLogisticTrainer


def load_local_clients(config: ProjectConfig) -> tuple[SharedPreprocessor, dict[str, tuple[object, object]]]:
    """Load each hospital training file independently without pooling rows."""
    first = pd.read_csv(config.data_dir / "hospital_A_train.csv", nrows=0)
    preprocessor = SharedPreprocessor.from_schema(first.columns)
    clients: dict[str, tuple[object, object]] = {}
    for hospital in config.hospitals:
        path = config.data_dir / f"hospital_{hospital}_train.csv"
        frame = pd.read_csv(path)
        clients[hospital] = (
            preprocessor.transform(frame),
            frame[config.target].astype(int).to_numpy(),
        )
    return preprocessor, clients


def main() -> None:
    """Train separate hospital clients and persist the global model artifact."""
    config = ProjectConfig()
    preprocessor, clients = load_local_clients(config)
    trainer = FederatedLogisticTrainer(
        config.rounds, config.max_iter, config.random_seed
    ).fit(clients, trust=True)
    history = pd.DataFrame(trainer.history_)
    history.to_csv(config.history_artifact, index=False)
    joblib.dump(
        {"trainer": trainer, "preprocessor": preprocessor, "trained_at": pd.Timestamp.utcnow().isoformat()},
        config.model_artifact,
    )
    print(f"Saved federated model: {config.model_artifact}")
    print(f"Saved training history: {config.history_artifact}")
    print("Final round:", trainer.history_[-1])


if __name__ == "__main__":
    main()
