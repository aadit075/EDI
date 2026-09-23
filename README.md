# FedMedTrust

FedMedTrust is a CPU-compatible reference implementation of federated cardiac-risk modeling across the five supplied hospital datasets. The implementation is designed to demonstrate the privacy boundary: hospital rows are loaded, transformed, and trained independently; patient records are never concatenated or sent to an aggregator.

## Data

Expected files are under `Data/`: `hospital_A_train.csv`, `hospital_A_validation.csv`, and `hospital_A_test.csv` (similarly B-E). The supplied `synthetic_cardiac_90000.csv` and hospital files are synthetic/demo data and must not be treated as real clinical evidence.

## Run

```text
pip install -r requirements.txt
python run_fedmedtrust.py
```

The script writes `federated_training_history.csv`. The original `train_federated.py` remains as a compatibility entry point, while the new package is in `fedmedtrust/`.

## Components

* `SharedPreprocessor` uses schema metadata and fixed category vocabularies; it does not fit pooled patient statistics.
* `FederatedLogisticTrainer` supports sample-weighted FedAvg and adaptive distance/trust aggregation.
* `FederatedXGBoostStrategy` uses an ensemble of complete local boosters, never an invalid average of tree coefficients. It gracefully degrades when XGBoost is unavailable.
* `HospitalStore` provides one SQLite database per hospital for insertion and prediction persistence.
* `create_app()` exposes dashboard pages and `/api/health`, `/api/hospitals`, and local patient insertion APIs.
* SHAP is optional; coefficient-based contributions are used as a fallback.

## Dashboard

```text
python -c "from fedmedtrust.dashboard import create_app; create_app().run()"
```

This is a research/demo implementation. It has no clinical validation, authentication, authorization, audit controls, or production privacy guarantees.
