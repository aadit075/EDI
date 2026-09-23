"""Record later-confirmed outcomes for stored FedMedTrust patients.

Predictions are written at intake time; the true outcome usually becomes
available afterwards. This helper updates only the confirmed_outcome field of
an existing hospital-local row, so it never rewrites features or predictions.

Usage:
    python record_outcome.py                       # interactive
    python record_outcome.py C P-001 1             # hospital, patient, outcome
    python record_outcome.py C P-001 unknown       # clear the outcome
    python record_outcome.py --list C              # recent local records
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from database import (
    DEFAULT_STORE_DIR,
    HOSPITALS,
    DatabaseError,
    count_patients,
    get_patient,
    list_patients,
    set_confirmed_outcome,
)

DISCLAIMER = (
    "Academic prototype using synthetic data; not a medical diagnosis. "
    "Outcomes recorded here are for prototype evaluation only."
)


class OutcomeError(RuntimeError):
    """Raised for invalid outcome input or missing patients."""


def _parse_outcome(raw: str) -> int | None:
    """Convert user outcome text into 0, 1, or None."""
    text = str(raw).strip().casefold()
    if text in {"", "unknown", "none", "blank", "-"}:
        return None
    if text in {"0", "no", "negative", "n"}:
        return 0
    if text in {"1", "yes", "positive", "y"}:
        return 1
    raise OutcomeError(
        f"Invalid outcome '{raw}'. Enter 0/1, or 'unknown' to clear the field."
    )


def _prompt(text: str) -> str:
    """Prompt for a required value."""
    value = input(text).strip()
    if not value:
        raise OutcomeError("A value is required.")
    return value


def _list(hospital: str, store_dir: Path) -> None:
    """Print recent local records without leaving the hospital database."""
    records = list_patients(hospital, limit=20, store_dir=store_dir)
    print(f"\nRecent records in hospital {hospital} "
          f"({count_patients(hospital, store_dir)} total):\n")
    if not records:
        print("  No records stored for this hospital yet.")
        return
    header = f"  {'Patient ID':<18}{'Risk':>8}{'Label':>7}{'Outcome':>9}{'Created':>22}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for record in records:
        probability = record.get("prediction")
        risk = f"{probability * 100:.1f}%" if isinstance(probability, (int, float)) else "-"
        label = record.get("predicted_label")
        outcome = record.get("confirmed_outcome")
        print(
            f"  {str(record.get('patient_id', '')):<18}{risk:>8}"
            f"{'-' if label is None else label:>7}"
            f"{'-' if outcome is None else outcome:>9}"
            f"{str(record.get('created_at', '')):>22}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Record or clear one confirmed outcome."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    print("FedMedTrust outcome recording")
    print(DISCLAIMER + "\n")

    if arguments and arguments[0] == "--list":
        if len(arguments) < 2:
            print("ERROR: --list requires a hospital letter.", file=sys.stderr)
            return 2
        try:
            _list(arguments[1].strip().upper(), DEFAULT_STORE_DIR)
        except DatabaseError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 1
        return 0

    try:
        if len(arguments) >= 3:
            hospital, patient_id, outcome_text = arguments[0], arguments[1], arguments[2]
        else:
            choices = "/".join(HOSPITALS)
            hospital = _prompt(f"Select hospital branch ({choices}): ")
            patient_id = _prompt("Patient ID: ")
            outcome_text = input("Confirmed outcome (0/1, 'unknown' to clear): ").strip()
        outcome = _parse_outcome(outcome_text)
        hospital_id = hospital.strip().upper()
        existing = get_patient(hospital_id, patient_id)
        if existing is None:
            raise OutcomeError(
                f"Patient '{patient_id}' is not in hospital {hospital_id}'s database."
            )
        set_confirmed_outcome(hospital_id, patient_id, outcome)
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled; nothing was changed.")
        return 130
    except (DatabaseError, OutcomeError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1

    probability = existing.get("prediction")
    label = existing.get("predicted_label")
    print(f"\n[OK] Hospital {hospital_id} patient {patient_id}")
    print(f"     Predicted risk : "
          f"{'not yet scored' if probability is None else f'{probability * 100:.2f}%'}")
    print(f"     Predicted label: {'-' if label is None else label}")
    print(f"     Confirmed now  : {'cleared' if outcome is None else outcome}")
    if outcome is not None and label is not None:
        verdict = "matches" if int(outcome) == int(label) else "differs from"
        print(f"     Outcome {verdict} the stored prediction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
