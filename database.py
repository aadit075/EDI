"""Per-hospital SQLite storage for the FedMedTrust academic prototype.

Each hospital owns exactly one SQLite file. Patient rows, predictions, and
confirmed outcomes are written only to that hospital's file; there is no
central or shared patient database. Raw patient data never leaves a hospital.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

HOSPITALS: tuple[str, ...] = ("A", "B", "C", "D", "E")
ROOT = Path(__file__).resolve().parent
DEFAULT_STORE_DIR = ROOT / "hospital_stores"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
TARGET = "cardiac_risk_label"

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id          TEXT PRIMARY KEY,
    hospital_id         TEXT NOT NULL,
    features            TEXT NOT NULL,
    prediction          REAL,
    predicted_label     INTEGER,
    confirmed_outcome   INTEGER,
    created_at          TEXT NOT NULL,
    predicted_at        TEXT
);

CREATE INDEX IF NOT EXISTS idx_patients_created_at ON patients (created_at);
"""

EXPECTED_COLUMNS = (
    "patient_id",
    "hospital_id",
    "features",
    "prediction",
    "predicted_label",
    "confirmed_outcome",
    "created_at",
    "predicted_at",
)


class DatabaseError(RuntimeError):
    """Raised when a hospital database cannot be opened or written."""


def database_path(hospital: str, store_dir: Path = DEFAULT_STORE_DIR) -> Path:
    """Return the SQLite file path for one hospital."""
    return Path(store_dir) / f"hospital_{validate_hospital(hospital)}.sqlite"


def validate_hospital(hospital: str) -> str:
    """Normalize and validate a hospital identifier."""
    normalized = str(hospital).strip().upper()
    if normalized not in HOSPITALS:
        raise DatabaseError(
            f"Unknown hospital '{hospital}'. Choose one of: {', '.join(HOSPITALS)}."
        )
    return normalized


def _utc_now() -> str:
    """Return a UTC timestamp string used for record bookkeeping."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _connect(path: Path) -> sqlite3.Connection:
    """Open one hospital-local database, creating its parent directory."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def _table_columns(connection: sqlite3.Connection) -> list[str]:
    """Return the current column names of the local patients table."""
    return [str(row[1]) for row in connection.execute("PRAGMA table_info(patients)").fetchall()]


def _migrate_legacy_schema(connection: sqlite3.Connection, hospital_id: str) -> bool:
    """Upgrade a legacy patients table in place, preserving existing rows.

    Earlier builds stored only (patient_id, payload, prediction, created_at).
    Legacy rows keep their prediction and timestamp; the payload is carried
    into the new features column so no prior record is discarded.
    """
    existing = set(_table_columns(connection))
    if "payload" not in existing:
        return False
    if {"hospital_id", "features", "predicted_label"} <= existing:
        return False

    connection.execute("ALTER TABLE patients RENAME TO patients_legacy")
    connection.executescript(SCHEMA)
    connection.execute(
        """
        INSERT INTO patients
            (patient_id, hospital_id, features, prediction, predicted_label,
             confirmed_outcome, created_at, predicted_at)
        SELECT
            patient_id,
            ?,
            COALESCE(payload, '{}'),
            prediction,
            CASE
                WHEN prediction IS NULL THEN NULL
                WHEN prediction >= 0.5 THEN 1
                ELSE 0
            END,
            NULL,
            COALESCE(created_at, ?),
            NULL
         FROM patients_legacy
        """,
        (hospital_id, _utc_now()),
    )
    connection.execute("DROP TABLE patients_legacy")
    return True


def initialize_database(hospital: str, store_dir: Path = DEFAULT_STORE_DIR) -> Path:
    """Create one hospital's database and table, migrating legacy files."""
    hospital_id = validate_hospital(hospital)
    path = database_path(hospital_id, store_dir)
    try:
        with _connect(path) as connection:
            connection.executescript(SCHEMA)
            if _migrate_legacy_schema(connection, hospital_id):
                print(f"[i] Upgraded legacy schema in {path.name} (hospital {hospital_id}).")
    except sqlite3.Error as error:
        raise DatabaseError(f"Could not initialize database '{path}': {error}") from error
    return path


def initialize_all_databases(store_dir: Path = DEFAULT_STORE_DIR) -> list[Path]:
    """Create all five per-hospital databases and return their paths."""
    return [initialize_database(hospital, store_dir) for hospital in HOSPITALS]


def _validate_features(features: Mapping[str, Any]) -> dict[str, Any]:
    """Validate that a patient record is a non-empty mapping of scalar values."""
    if not isinstance(features, Mapping):
        raise DatabaseError("Patient features must be a mapping of field names to values.")
    cleaned: dict[str, Any] = {}
    for key, value in features.items():
        name = str(key).strip()
        if not name:
            raise DatabaseError("Patient feature names cannot be empty.")
        if isinstance(value, (dict, list, tuple, set)):
            raise DatabaseError(f"Feature '{name}' must be a single scalar value.")
        cleaned[name] = value
    if not cleaned:
        raise DatabaseError("Patient record is empty; at least one feature is required.")
    return cleaned


def save_patient(
    hospital: str,
    patient_id: str,
    features: Mapping[str, Any],
    store_dir: Path = DEFAULT_STORE_DIR,
) -> dict[str, Any]:
    """Insert one patient row into that hospital's database, before prediction."""
    hospital_id = validate_hospital(hospital)
    clean_id = str(patient_id).strip()
    if not clean_id:
        raise DatabaseError("patient_id cannot be empty.")
    record = _validate_features(features)
    path = initialize_database(hospital_id, store_dir)
    created_at = _utc_now()
    try:
        with _connect(path) as connection:
            connection.execute(
                """
                INSERT INTO patients
                    (patient_id, hospital_id, features, prediction, predicted_label,
                     confirmed_outcome, created_at, predicted_at)
                VALUES (?, ?, ?, NULL, NULL, NULL, ?, NULL)
                """,
                (clean_id, hospital_id, json.dumps(record, default=str), created_at),
            )
    except sqlite3.IntegrityError as error:
        raise DatabaseError(
            f"patient_id '{clean_id}' already exists in hospital {hospital_id}'s database."
        ) from error
    except sqlite3.Error as error:
        raise DatabaseError(f"Could not save patient '{clean_id}': {error}") from error
    return {"patient_id": clean_id, "hospital_id": hospital_id, "created_at": created_at}


def update_prediction(
    hospital: str,
    patient_id: str,
    probability: float,
    label: int | None = None,
    threshold: float = 0.5,
    store_dir: Path = DEFAULT_STORE_DIR,
) -> dict[str, Any]:
    """Write the model prediction back to the same hospital-local patient row."""
    hospital_id = validate_hospital(hospital)
    clean_id = str(patient_id).strip()
    if not clean_id:
        raise DatabaseError("patient_id cannot be empty.")
    try:
        bounded = float(probability)
    except (TypeError, ValueError) as error:
        raise DatabaseError(f"Prediction probability must be numeric, received {probability!r}.") from error
    if not 0.0 <= bounded <= 1.0:
        raise DatabaseError(f"Prediction probability must be between 0 and 1, received {bounded}.")
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError) as error:
        raise DatabaseError(f"Threshold must be numeric, received {threshold!r}.") from error
    predicted_label = int(bounded >= threshold_value) if label is None else int(label)
    if predicted_label not in (0, 1):
        raise DatabaseError("Predicted label must be 0 (lower risk) or 1 (higher risk).")
    path = database_path(hospital_id, store_dir)
    predicted_at = _utc_now()
    try:
        with _connect(path) as connection:
            cursor = connection.execute(
                """
                UPDATE patients
                   SET prediction = ?, predicted_label = ?, predicted_at = ?
                 WHERE patient_id = ? AND hospital_id = ?
                """,
                (bounded, predicted_label, predicted_at, clean_id, hospital_id),
            )
            if cursor.rowcount == 0:
                raise DatabaseError(
                    f"Patient '{clean_id}' was not found in hospital {hospital_id}'s database."
                )
    except sqlite3.Error as error:
        raise DatabaseError(f"Could not update prediction for '{clean_id}': {error}") from error
    return {
        "patient_id": clean_id,
        "hospital_id": hospital_id,
        "prediction": bounded,
        "predicted_label": predicted_label,
        "predicted_at": predicted_at,
    }


def set_confirmed_outcome(
    hospital: str,
    patient_id: str,
    outcome: int | None,
    store_dir: Path = DEFAULT_STORE_DIR,
) -> dict[str, Any]:
    """Record or clear the later-confirmed outcome for one local patient."""
    hospital_id = validate_hospital(hospital)
    clean_id = str(patient_id).strip()
    if not clean_id:
        raise DatabaseError("patient_id cannot be empty.")
    if outcome is None:
        value: int | None = None
    else:
        if isinstance(outcome, bool) or str(outcome).strip() not in {"0", "1"}:
            raise DatabaseError("Confirmed outcome must be 0, 1, or blank for unknown.")
        value = int(str(outcome).strip())
    path = database_path(hospital_id, store_dir)
    try:
        with _connect(path) as connection:
            cursor = connection.execute(
                "UPDATE patients SET confirmed_outcome = ? WHERE patient_id = ? AND hospital_id = ?",
                (value, clean_id, hospital_id),
            )
            if cursor.rowcount == 0:
                raise DatabaseError(
                    f"Patient '{clean_id}' was not found in hospital {hospital_id}'s database."
                )
    except sqlite3.Error as error:
        raise DatabaseError(f"Could not store confirmed outcome for '{clean_id}': {error}") from error
    return {"patient_id": clean_id, "hospital_id": hospital_id, "confirmed_outcome": value}


def _row_to_record(row: sqlite3.Row) -> dict[str, Any]:
    """Convert one SQLite row into a plain dictionary."""
    record = dict(row)
    raw = record.get("features")
    if isinstance(raw, str):
        try:
            record["features"] = json.loads(raw)
        except json.JSONDecodeError:
            record["features"] = {}
    return record


def get_patient(
    hospital: str,
    patient_id: str,
    store_dir: Path = DEFAULT_STORE_DIR,
) -> dict[str, Any] | None:
    """Return one local patient record, or None when it is absent."""
    hospital_id = validate_hospital(hospital)
    clean_id = str(patient_id).strip()
    if not clean_id:
        raise DatabaseError("patient_id cannot be empty.")
    path = database_path(hospital_id, store_dir)
    if not path.is_file():
        return None
    try:
        with _connect(path) as connection:
            row = connection.execute(
                "SELECT * FROM patients WHERE patient_id = ? AND hospital_id = ?",
                (clean_id, hospital_id),
            ).fetchone()
    except sqlite3.Error as error:
        raise DatabaseError(f"Could not read patient '{clean_id}': {error}") from error
    return _row_to_record(row) if row is not None else None


def list_patients(
    hospital: str,
    limit: int = 20,
    store_dir: Path = DEFAULT_STORE_DIR,
) -> list[dict[str, Any]]:
    """List recent local records for one hospital only."""
    hospital_id = validate_hospital(hospital)
    safe_limit = max(1, min(int(limit), 500))
    path = database_path(hospital_id, store_dir)
    if not path.is_file():
        return []
    try:
        with _connect(path) as connection:
            rows = connection.execute(
                """
                SELECT patient_id, hospital_id, prediction, predicted_label,
                       confirmed_outcome, created_at, predicted_at
                  FROM patients
                 WHERE hospital_id = ?
                 ORDER BY created_at DESC, rowid DESC
                 LIMIT ?
                """,
                (hospital_id, safe_limit),
            ).fetchall()
    except sqlite3.Error as error:
        raise DatabaseError(f"Could not list patients for hospital {hospital_id}: {error}") from error
    return [dict(row) for row in rows]


def count_patients(hospital: str, store_dir: Path = DEFAULT_STORE_DIR) -> int:
    """Count local records stored for one hospital."""
    hospital_id = validate_hospital(hospital)
    path = database_path(hospital_id, store_dir)
    if not path.is_file():
        return 0
    try:
        with _connect(path) as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM patients WHERE hospital_id = ?", (hospital_id,)
            ).fetchone()
    except sqlite3.Error as error:
        raise DatabaseError(f"Could not count patients for hospital {hospital_id}: {error}") from error
    return int(row["total"] if row else 0)


def _print_summary(store_dir: Path) -> None:
    """Print one line per hospital database without merging their contents."""
    print(f"Store directory: {Path(store_dir).resolve()}\n")
    print(f"{'Hospital':<10}{'Database':<34}{'Patients':>9}")
    print("-" * 53)
    for hospital in HOSPITALS:
        path = database_path(hospital, store_dir)
        print(f"{hospital:<10}{path.name:<34}{count_patients(hospital, store_dir):>9}")
    print(
        "\nEach hospital keeps its own file. No central patient database is created."
    )


def _print_patients(hospital: str, store_dir: Path) -> None:
    """Print recent local records for one hospital."""
    records = list_patients(hospital, limit=20, store_dir=store_dir)
    print(f"Recent local records for hospital {hospital}:\n")
    if not records:
        print("  No records stored for this hospital yet.")
        return
    header = f"{'Patient ID':<18}{'Risk':>8}{'Label':>7}{'Outcome':>9}{'Created':>22}"
    print(header)
    print("-" * len(header))
    for record in records:
        probability = record.get("prediction")
        risk = f"{probability * 100:.1f}%" if isinstance(probability, (int, float)) else "-"
        label = record.get("predicted_label")
        label_text = "-" if label is None else str(label)
        outcome = record.get("confirmed_outcome")
        outcome_text = "-" if outcome is None else str(outcome)
        print(
            f"{str(record.get('patient_id', '')):<18}{risk:>8}{label_text:>7}"
            f"{outcome_text:>9}{str(record.get('created_at', '')):>22}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Initialize the five hospital databases and optionally inspect them."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    store_dir = DEFAULT_STORE_DIR
    if arguments and not arguments[0].startswith("-"):
        store_dir = Path(arguments[0])

    print("FedMedTrust local hospital databases")
    print("Academic prototype using synthetic data; not a medical diagnosis.\n")
    try:
        paths = initialize_all_databases(store_dir)
    except DatabaseError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    for path in paths:
        print(f"[OK] Initialized {path}")
    print()
    _print_summary(store_dir)

    if "--list" in arguments and len(arguments) > 1:
        target = arguments[1] if arguments[0].startswith("-") else None
        if target:
            try:
                _print_patients(validate_hospital(target), store_dir)
            except DatabaseError as error:
                print(f"ERROR: {error}", file=sys.stderr)
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
