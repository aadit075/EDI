"""Local SQLite hospital storage."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union


class HospitalStore:
    """Stores patient records and predictions in one hospital-local SQLite file."""

    def __init__(self, path: Path) -> None:
        """Initialize the database and create its local patient table."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS patients (patient_id TEXT PRIMARY KEY, payload TEXT NOT NULL, prediction REAL, created_at TEXT)")
            columns = {row[1] for row in connection.execute("PRAGMA table_info(patients)").fetchall()}
            # Migrate: database.py used 'features'; store.py uses 'payload'
            if "payload" not in columns and "features" in columns:
                connection.execute("ALTER TABLE patients RENAME COLUMN features TO payload")
                columns.add("payload")
            if "created_at" not in columns:
                connection.execute("ALTER TABLE patients ADD COLUMN created_at TEXT")
            if "xray_predictions" not in columns:
                connection.execute("ALTER TABLE patients ADD COLUMN xray_predictions TEXT")

    def insert_patient(self, patient_id: str, record: Mapping[str, Any]) -> None:
        """Insert one patient record, rejecting empty or duplicate identifiers."""
        clean_id = str(patient_id).strip()
        if not clean_id:
            raise ValueError("patient_id cannot be empty")
        payload = json.dumps(dict(record), default=str)
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        with sqlite3.connect(self.path) as connection:
            try:
                connection.execute("INSERT INTO patients(patient_id,payload,prediction,created_at) VALUES(?,?,NULL,?)", (clean_id, payload, created_at))
            except sqlite3.IntegrityError as error:
                raise ValueError("patient_id already exists") from error

    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Return one local patient record, or None when absent."""
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT payload,prediction,created_at,xray_predictions FROM patients WHERE patient_id=?",
                (patient_id,),
            ).fetchone()
        if row is None:
            return None
        result = json.loads(row[0])
        result["patient_id"] = patient_id
        result["prediction"] = row[1]
        result["created_at"] = row[2] or "Unknown"
        result["xray_predictions"] = json.loads(row[3]) if row[3] else None
        return result

    def set_prediction(self, patient_id: str, probability: float) -> None:
        """Persist a bounded model probability for a locally stored patient."""
        bounded = min(1.0, max(0.0, float(probability)))
        with sqlite3.connect(self.path) as connection:
            connection.execute("UPDATE patients SET prediction=? WHERE patient_id=?", (bounded, patient_id))

    def set_xray_predictions(self, patient_id: str, predictions: Dict[str, Any]) -> None:
        """Persist X-ray finding probabilities as JSON for a locally stored patient."""
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "UPDATE patients SET xray_predictions=? WHERE patient_id=?",
                (json.dumps(predictions), patient_id),
            )

    def count(self) -> int:
        """Return the number of records in this hospital's database."""
        with sqlite3.connect(self.path) as connection:
            row = connection.execute("SELECT COUNT(*) FROM patients").fetchone()
        return int(row[0] if row else 0)

    def list_patients(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return a bounded summary list without combining hospital databases."""
        safe_limit = max(1, min(int(limit), 500))
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute("SELECT patient_id,prediction,created_at FROM patients ORDER BY rowid DESC LIMIT ?", (safe_limit,)).fetchall()
        return [{"patient_id": row[0], "prediction": row[1], "created_at": row[2] or "Unknown"} for row in rows]