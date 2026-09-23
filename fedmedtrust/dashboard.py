"""Polished local-only FedMedTrust hospital network portal."""
from __future__ import annotations

import base64
import html
import math
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import joblib
import pandas as pd
from flask import Flask, Response, flash, jsonify, redirect, render_template_string, request, session, url_for

from .config import ProjectConfig
from .store import HospitalStore

try:
    from fedmedtrust_images.predict_image import predict_xray_from_bytes, DISEASE_LABELS as XRAY_LABELS
    _XRAY_AVAILABLE = True
except Exception:  # torch / torchvision not installed
    _XRAY_AVAILABLE = False
    XRAY_LABELS = ()

NETWORK_NAME = "HeartShield Network"
BRANCHES: Dict[str, Dict[str, str]] = {
    "A": {"name": "Northside Cardiac Center", "city": "North District"},
    "B": {"name": "Riverside Medical Pavilion", "city": "River District"},
    "C": {"name": "Summit Heart Institute", "city": "Summit District"},
    "D": {"name": "Lakeside General Hospital", "city": "Lake District"},
    "E": {"name": "Eastgate Health Campus", "city": "East District"},
}
DEMO_CREDENTIALS: Dict[str, Dict[str, str]] = {
    "A": {"username": "northside", "password": "demo-A"},
    "B": {"username": "riverside", "password": "demo-B"},
    "C": {"username": "summit", "password": "demo-C"},
    "D": {"username": "lakeside", "password": "demo-D"},
    "E": {"username": "eastgate", "password": "demo-E"},
}
FIELDS: List[tuple[str, str, str, str]] = [
    ("age", "Age", "number", "Years"), ("sex", "Sex", "select", ""),
    ("bmi", "BMI", "number", "kg/m²"), ("heart_rate", "Heart rate", "number", "bpm"),
    ("systolic_bp", "Systolic BP", "number", "mmHg"), ("diastolic_bp", "Diastolic BP", "number", "mmHg"),
    ("oxygen_saturation", "Oxygen saturation", "number", "%"), ("diabetes", "Diabetes", "select", ""),
    ("hypertension", "Hypertension", "select", ""), ("coronary_artery_disease", "Coronary artery disease", "select", ""),
    ("smoking_status", "Smoking status", "select", ""), ("family_history_cardiac_disease", "Family cardiac history", "select", ""),
    ("physical_activity_level", "Physical activity", "select", ""), ("chest_pain", "Chest pain", "select", ""),
    ("shortness_of_breath", "Shortness of breath", "select", ""), ("fatigue", "Fatigue", "select", ""),
    ("palpitations", "Palpitations", "select", ""), ("total_cholesterol", "Total cholesterol", "number", "mg/dL"),
    ("ldl_cholesterol", "LDL cholesterol", "number", "mg/dL"), ("hdl_cholesterol", "HDL cholesterol", "number", "mg/dL"),
    ("triglycerides", "Triglycerides", "number", "mg/dL"), ("blood_glucose", "Blood glucose", "number", "mg/dL"),
    ("hba1c", "HbA1c", "number", "%"), ("creatinine", "Creatinine", "number", "mg/dL"),
]
SELECTS: Dict[str, List[tuple[str, str]]] = {
    "sex": [("Female", "Female"), ("Male", "Male"), ("Other", "Other")],
    "smoking_status": [("never", "Never"), ("former", "Former"), ("current", "Current")],
    "physical_activity_level": [("low", "Low"), ("moderate", "Moderate"), ("high", "High")],
}
for _name in ("diabetes", "hypertension", "coronary_artery_disease", "family_history_cardiac_disease", "chest_pain", "shortness_of_breath", "fatigue", "palpitations"):
    SELECTS[_name] = [("0", "No"), ("1", "Yes")]

SHELL = """<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{{ title }} · HeartShield</title><style>
:root{--ink:#10233f;--muted:#64748b;--blue:#1769e0;--cyan:#15b8a6;--bg:#f4f7fb;--line:#e5ebf3;--card:#fff;--red:#e05252}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.app{display:flex;min-height:100vh}.side{width:248px;background:#0d2341;color:#d9e6fa;padding:25px 18px;display:flex;flex-direction:column}.brand{font-size:20px;font-weight:800;color:white;margin:0 10px 36px;letter-spacing:-.4px}.brand span{color:#49d5c4}.nav a{display:block;color:#a9bfdc;text-decoration:none;padding:12px 13px;border-radius:10px;margin:4px 0}.nav a:hover,.nav .active{background:#183c68;color:white}.side small{margin-top:auto;color:#7f9abc;line-height:1.5;padding:10px}.main{flex:1;max-width:1500px}.top{height:74px;background:white;border-bottom:1px solid var(--line);padding:0 38px;display:flex;justify-content:space-between;align-items:center}.top h1{font-size:21px;margin:0}.who{color:var(--muted)}.content{padding:34px 38px;max-width:1240px}.eyebrow{color:var(--blue);font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:1.3px}.muted{color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:22px;box-shadow:0 5px 18px #153d6910}.stat{font-size:31px;font-weight:800;margin:8px 0}.stat-label{color:var(--muted)}.hero{background:linear-gradient(120deg,#10355e,#1876c9);color:white;border-radius:18px;padding:30px;margin-bottom:22px}.hero h2{font-size:28px;margin:5px 0 10px}.hero p{color:#d6eaff;max-width:650px}.btn{border:0;border-radius:9px;background:var(--blue);color:white;padding:11px 17px;font-weight:700;text-decoration:none;display:inline-block;cursor:pointer}.btn.secondary{background:#eaf2ff;color:#1555af}.btn.ghost{background:transparent;color:#eaf2ff;border:1px solid #6f9bd0}.row{display:flex;align-items:center;justify-content:space-between;gap:15px}.section-title{margin:0 0 16px;font-size:18px}.notice{padding:13px 16px;background:#fff8e6;color:#805b00;border:1px solid #f5df9d;border-radius:10px;margin:14px 0}.form-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:17px}.field label{display:block;font-size:12px;font-weight:750;margin-bottom:7px;color:#44546a}.field input,.field select{width:100%;border:1px solid #d5deea;border-radius:8px;background:white;padding:11px;color:var(--ink);font-size:14px}.field input:focus,.field select:focus{outline:2px solid #bcd7ff;border-color:var(--blue)}.actions{margin-top:24px;display:flex;gap:10px;align-items:center}.risk{font-size:44px;font-weight:850;color:var(--blue)}.bar{height:10px;background:#e7eef8;border-radius:9px;overflow:hidden}.bar i{display:block;height:100%;background:linear-gradient(90deg,#19b99c,#f0bb4c,#de5f59);border-radius:9px}.factors{display:grid;gap:10px}.factor{display:grid;grid-template-columns:170px 1fr 55px;gap:10px;align-items:center;color:#42536b}.table-wrap{overflow:auto}.table{width:100%;border-collapse:collapse}.table th,.table td{text-align:left;padding:13px 10px;border-bottom:1px solid var(--line)}.table th{font-size:12px;text-transform:uppercase;color:var(--muted)}.pill{border-radius:20px;padding:5px 10px;font-size:12px;font-weight:700;background:#e9f8f5;color:#087c6d}.pill.high{background:#fff0ed;color:#b7473d}.login{min-height:100vh;display:grid;place-items:center;background:radial-gradient(circle at 80% 10%,#2865a8,#0b2343 55%);padding:20px}.login-box{width:min(500px,100%);background:white;border-radius:20px;padding:36px;box-shadow:0 20px 60px #0004}.branch-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:18px 0}.branch{border:1px solid var(--line);padding:10px 4px;text-align:center;border-radius:9px;cursor:pointer;background:#fafcff}.branch.selected{border-color:var(--blue);background:#eaf2ff;color:var(--blue)}.error{color:var(--red);margin:10px 0}.fine{font-size:12px;color:var(--muted)}@media(max-width:850px){.side{width:70px;padding:20px 8px}.brand{font-size:0;margin:0 10px 30px}.brand span{font-size:20px}.nav a{font-size:0;text-align:center}.nav a:first-letter{font-size:18px}.side small{display:none}.content,.top{padding-left:20px;padding-right:20px}.grid,.form-grid{grid-template-columns:1fr 1fr}}@media(max-width:560px){.grid,.form-grid{grid-template-columns:1fr}.top h1{font-size:16px}.who{font-size:12px}.content{padding-top:22px}}
.xray-upload{border:2px dashed var(--line);border-radius:12px;padding:22px;text-align:center;background:#fafcff;cursor:pointer;transition:border-color .2s}.xray-upload:hover{border-color:var(--blue)}.xray-upload input{display:none}.xray-upload b{display:block;color:var(--ink);margin-top:8px}.xray-upload small{color:var(--muted)}.xray-filename{margin-top:8px;color:var(--blue);font-size:13px;font-weight:600}.finding-row{display:grid;grid-template-columns:160px 1fr 56px;gap:10px;align-items:center;margin:6px 0}.finding-label{font-size:13px;color:#42536b}.finding-bar{height:8px;background:#e7eef8;border-radius:6px;overflow:hidden}.finding-fill{height:100%;border-radius:6px;transition:width .3s}.finding-pct{font-size:12px;font-weight:700;text-align:right}.badge-high{background:#fff0ed;color:#b7473d;border-radius:4px;padding:1px 5px;font-size:11px;font-weight:700}.badge-mod{background:#fff8e6;color:#805b00;border-radius:4px;padding:1px 5px;font-size:11px;font-weight:700}.xray-section{margin-top:22px}.xray-highlight{background:linear-gradient(135deg,#eaf7f5,#f0f7ff);border:1px solid #c5e8e4;border-radius:12px;padding:16px 20px;margin-bottom:14px}
</style></head><body>{{ body|safe }}</body></html>"""

LOGIN = """<div class='login'><div class='login-box'><div class='eyebrow'>Private clinical workspace</div><h1 style='font-size:30px;margin:8px 0'>HeartShield <span style='color:#15b8a6'>Network</span></h1><p class='muted'>Federated cardiac-risk demonstration portal for hospital teams.</p>{% with messages=get_flashed_messages() %}{% if messages %}<div class='error'>{{ messages[0] }}</div>{% endif %}{% endwith %}<form method='post'><label class='field'><span style='display:block;margin-bottom:7px;font-weight:700'>Choose branch</span><div class='branch-grid'>{% for key,branch in branches.items() %}<label class='branch {% if request.form.get("hospital", "A") == key %}selected{% endif %}'><input type='radio' name='hospital' value='{{ key }}' {% if request.form.get("hospital", "A") == key %}checked{% endif %} style='display:none'> <b>{{ key }}</b><br><small>{{ branch.name.split(" ")[0] }}</small></label>{% endfor %}</div></label><div class='field' style='margin-top:18px'><label>Username</label><input name='username' required autocomplete='username' value='{{ request.form.get("username", "") }}'></div><div class='field' style='margin-top:14px'><label>Password</label><input type='password' name='password' required autocomplete='current-password'></div><button class='btn' style='width:100%;margin-top:22px'>Sign in to branch</button></form><p class='fine' style='margin-top:20px'>Demo only · credentials are shown in the project documentation. No real patient data.</p></div></div>"""


def _page(title: str, body_template: str, **context: Any) -> str:
    """Render a portal page inside the shared shell."""
    body = render_template_string(body_template, title=title, **context)
    return render_template_string(SHELL, title=title, body=body)


CLINICAL_DEFAULTS: Dict[str, Any] = {
    "age": 52.0, "sex": "Unknown", "height_cm": 168.0, "weight_kg": 72.0, "bmi": 26.5,
    "heart_rate": 74.0, "systolic_bp": 130.0, "diastolic_bp": 82.0, "oxygen_saturation": 98.0,
    "respiratory_rate": 16.0, "body_temperature": 36.8, "diabetes": "0", "hypertension": "0",
    "coronary_artery_disease": "0", "smoking_status": "never", "family_history_cardiac_disease": "0",
    "physical_activity_level": "moderate", "chest_pain": "0", "shortness_of_breath": "0",
    "fatigue": "0", "palpitations": "0", "hemoglobin": 14.0, "white_blood_cell_count": 7.0,
    "creatinine": 0.9, "blood_glucose": 95.0, "hba1c": 5.4, "total_cholesterol": 190.0,
    "ldl_cholesterol": 105.0, "hdl_cholesterol": 52.0, "triglycerides": 135.0, "sodium": 139.0,
    "potassium": 4.2, "cardiac_biomarker": 30.0, "previous_cardiac_hospitalization": "0",
}


def _prepare_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge submitted form values with clinical baseline defaults for unsubmitted fields."""
    merged = dict(CLINICAL_DEFAULTS)
    for k, v in record.items():
        if v is not None and str(v).strip() != "":
            merged[k] = v
    return merged


def _probability(record: Mapping[str, Any]) -> float:
    """Compute a deterministic, transparent fallback probability from submitted values."""
    def number(name: str, default: float = 0.0) -> float:
        try:
            value = float(record.get(name, default))
            return value if math.isfinite(value) else default
        except (TypeError, ValueError):
            return default
    age = number("age", 45.0)
    bmi = number("bmi", 25.0)
    bp = number("systolic_bp", 125.0)
    chol = number("total_cholesterol", 190.0)
    gluc = number("blood_glucose", 95.0)
    score = -2.2 + max(-15, age - 25) * 0.035 + max(-5, bmi - 20) * 0.040
    score += max(-20, bp - 110) * 0.020 + max(-50, chol - 150) * 0.005 + max(-20, gluc - 85) * 0.006
    score += sum(0.55 for name in ("diabetes", "hypertension", "coronary_artery_disease") if str(record.get(name, "0")).lower() in ("1", "yes"))
    if str(record.get("smoking_status", "never")).lower() == "current":
        score += 0.45
    score += sum(0.30 for name in ("family_history_cardiac_disease", "chest_pain", "shortness_of_breath", "palpitations") if str(record.get(name, "0")) in ("1", "yes"))
    return round(1 / (1 + math.exp(-max(-6, min(6, score)))), 4)


def _factors(record: Mapping[str, Any], artifact: Optional[Dict[str, Any]] = None) -> List[tuple[str, float]]:
    """Return patient-specific risk factor contributions indicating clinical deviation from normal."""
    merged = _prepare_record(record)
    clinical_factors: List[tuple[str, float, float]] = []

    # 1. Active clinical conditions
    if str(record.get("hypertension", "0")).lower() in ("1", "yes"):
        clinical_factors.append(("Hypertension (Diagnosed)", 0.92, 1.40))
    if str(record.get("diabetes", "0")).lower() in ("1", "yes"):
        clinical_factors.append(("Diabetes (Metabolic Risk)", 0.82, 0.52))
    if str(record.get("coronary_artery_disease", "0")).lower() in ("1", "yes"):
        clinical_factors.append(("Coronary Artery Disease History", 0.85, 0.49))
    if str(record.get("chest_pain", "0")).lower() in ("1", "yes"):
        clinical_factors.append(("Chest Pain (Angina Symptoms)", 0.78, 0.37))
    if str(record.get("shortness_of_breath", "0")).lower() in ("1", "yes"):
        clinical_factors.append(("Shortness of Breath (Dyspnea)", 0.68, 0.31))
    if str(record.get("smoking_status", "never")).lower() == "current":
        clinical_factors.append(("Current Tobacco Smoking", 0.75, 0.60))
    elif str(record.get("smoking_status", "never")).lower() == "former":
        clinical_factors.append(("Former Smoker History", 0.35, 0.15))
    if str(record.get("family_history_cardiac_disease", "0")).lower() in ("1", "yes"):
        clinical_factors.append(("Family Cardiac History", 0.60, 0.28))
    if str(record.get("palpitations", "0")).lower() in ("1", "yes"):
        clinical_factors.append(("Reported Palpitations / Arrhythmia", 0.55, 0.20))

    # 2. Continuous clinical metrics (only flagged when deviating from normal)
    def num(key: str, default: float) -> float:
        try:
            v = record.get(key)
            return float(v) if v not in ("", None) else default
        except (ValueError, TypeError):
            return default

    # Blood Pressure (Optimal < 120, Elevated 120-129, Stage 1 130-139, Stage 2 >= 140)
    sbp = num("systolic_bp", 120.0)
    if sbp >= 140:
        sev = min(0.95, 0.60 + (sbp - 140) * 0.015)
        clinical_factors.append((f"High Blood Pressure ({int(sbp)} mmHg)", round(sev, 2), (sbp - 120) * 0.015))
    elif sbp >= 130:
        clinical_factors.append((f"Stage 1 Hypertension ({int(sbp)} mmHg)", 0.48, (sbp - 120) * 0.010))

    # Age (Risk increases notably after age 45-50)
    age = num("age", 40.0)
    if age >= 65:
        sev = min(0.95, 0.65 + (age - 65) * 0.015)
        clinical_factors.append((f"Advanced Age Risk ({int(age)} yrs)", round(sev, 2), (age - 45) * 0.020))
    elif age >= 50:
        sev = 0.40 + (age - 50) * 0.015
        clinical_factors.append((f"Age Risk Factor ({int(age)} yrs)", round(sev, 2), (age - 45) * 0.015))

    # Body Mass Index (Normal 18.5-24.9, Overweight 25-29.9, Obese >= 30)
    bmi = num("bmi", 23.5)
    if bmi >= 30.0:
        sev = min(0.88, 0.55 + (bmi - 30.0) * 0.03)
        clinical_factors.append((f"Obesity / Elevated BMI ({bmi:.1f} kg/m2)", round(sev, 2), (bmi - 25.0) * 0.02))
    elif bmi >= 26.5:
        clinical_factors.append((f"Overweight BMI ({bmi:.1f} kg/m2)", 0.38, (bmi - 25.0) * 0.01))

    # Cholesterol (Desirable < 200, Borderline 200-239, High >= 240)
    chol = num("total_cholesterol", 180.0)
    if chol >= 240:
        sev = min(0.90, 0.60 + (chol - 240) * 0.005)
        clinical_factors.append((f"High Total Cholesterol ({int(chol)} mg/dL)", round(sev, 2), (chol - 190) * 0.005))
    elif chol >= 200:
        clinical_factors.append((f"Borderline Cholesterol ({int(chol)} mg/dL)", 0.42, (chol - 190) * 0.003))

    # Blood Glucose (Normal < 100, Impaired 100-125, Diabetic >= 126)
    gluc = num("blood_glucose", 95.0)
    if gluc >= 126:
        sev = min(0.90, 0.55 + (gluc - 126) * 0.005)
        clinical_factors.append((f"Hyperglycemia ({int(gluc)} mg/dL)", round(sev, 2), (gluc - 100) * 0.006))
    elif gluc >= 105:
        clinical_factors.append((f"Elevated Fasting Glucose ({int(gluc)} mg/dL)", 0.35, (gluc - 100) * 0.003))

    # If any elevated clinical factors were found, sort by risk contribution weight
    if clinical_factors:
        clinical_factors.sort(key=lambda x: x[2], reverse=True)
        return [(name, score) for name, score, _ in clinical_factors[:5]]

    # If patient has all normal/optimal inputs, display their actual readings as healthy indicators
    return [
        (f"Age Profile ({int(age)} yrs - Lower Risk)", round(min(0.28, max(0.08, age / 160)), 2)),
        (f"Resting Blood Pressure ({int(sbp)} mmHg - Optimal)", 0.12),
        (f"Body Mass Index ({bmi:.1f} kg/m2 - Healthy)", 0.14),
        ("No Diagnosed Hypertension or Diabetes", 0.08),
    ]


def _load_trained_artifact(config: ProjectConfig) -> Optional[Dict[str, Any]]:
    """Load the model trained by train_federated.py, if available."""
    if not config.model_artifact.exists():
        return None
    try:
        artifact = joblib.load(config.model_artifact)
        return artifact if isinstance(artifact, dict) and "trainer" in artifact and "preprocessor" in artifact else None
    except (OSError, ValueError, EOFError, ImportError):
        return None


def _trained_probability(record: Mapping[str, Any], artifact: Optional[Dict[str, Any]]) -> Optional[float]:
    """Generate a probability from the persisted federated model."""
    if artifact is None:
        return None
    try:
        prepared = _prepare_record(record)
        frame = pd.DataFrame([prepared])
        features = artifact["preprocessor"].transform(frame)
        return float(artifact["trainer"].predict_proba(features)[0, 1])
    except Exception as err:
        print(f"Error in _trained_probability: {err}")
        return None


def create_app(config: Optional[ProjectConfig] = None) -> Flask:
    """Create the authenticated HeartShield Network Flask application."""
    cfg = config or ProjectConfig()
    app = Flask(__name__)
    app.secret_key = "fedmedtrust-demo-secret-change-me"
    artifact = _load_trained_artifact(cfg)
    stores: Dict[str, HospitalStore] = {h: HospitalStore(cfg.root / "hospital_stores" / f"hospital_{h}.sqlite") for h in cfg.hospitals}

    def logged_in() -> bool:
        """Return whether a branch session is active."""
        return session.get("hospital") in stores

    def current_hospital() -> str:
        """Return the authenticated branch key."""
        return str(session["hospital"])

    def require_login() -> Optional[Response]:
        """Redirect unauthenticated browser requests to login."""
        return None if logged_in() else redirect(url_for("login", next=request.path))

    @app.route("/login", methods=["GET", "POST"])
    def login() -> str | Response:
        """Authenticate a demo user into exactly one hospital branch."""
        if request.method == "POST":
            hospital = request.form.get("hospital", "")
            credentials = DEMO_CREDENTIALS.get(hospital, {})
            if request.form.get("username") == credentials.get("username") and request.form.get("password") == credentials.get("password"):
                session.clear(); session["hospital"] = hospital; session["username"] = credentials["username"]
                return redirect(request.args.get("next") or url_for("dashboard"))
            flash("That demo credential does not match the selected branch.")
        return _page("Sign in", LOGIN, branches=BRANCHES)

    @app.get("/logout")
    def logout() -> Response:
        """End the current branch session."""
        session.clear(); return redirect(url_for("login"))

    @app.get("/")
    def home() -> Response:
        """Route the portal root to the dashboard or login."""
        return redirect(url_for("dashboard"))

    @app.get("/dashboard")
    def dashboard() -> str | Response:
        """Render branch-scoped dashboard metrics."""
        denied = require_login()
        if denied: return denied
        hospital = current_hospital(); store = stores[hospital]
        return _page("Dashboard", """<div class='hero'><div class='eyebrow' style='color:#8ae5db'>{{ branch.city }} · Branch {{ hospital }}</div><h2>Good morning, {{ username }}.</h2><p>Monitor your local cardiac-risk workflow with privacy-preserving predictions. Records stay in {{ branch.name }}.</p><a class='btn ghost' href='{{ url_for("predict") }}'>＋ New assessment</a></div><div class='grid'><div class='card'><div class='stat'>{{ store.count() }}</div><div class='stat-label'>Local assessments</div></div><div class='card'><div class='stat'>{{ branch.name.split(" ")|length + 1 }}</div><div class='stat-label'>Secure branch workspace</div></div><div class='card'><div class='stat' style='color:#15a88f'>On</div><div class='stat-label'>Privacy boundary</div></div></div><div class='card' style='margin-top:22px'><div class='row'><h3 class='section-title'>Recent local assessments</h3><a class='btn secondary' href='{{ url_for("patients") }}'>View all</a></div>{% if patients %}<div class='table-wrap'><table class='table'><tr><th>Patient ID</th><th>Assessment</th><th>Risk</th><th></th></tr>{% for p in patients %}<tr><td><b>{{ p.patient_id }}</b></td><td>{{ p.created_at }}</td><td><span class='pill {% if p.prediction and p.prediction >= 0.5 %}high{% endif %}'>{{ "%.0f"|format((p.prediction or 0)*100) }}%</span></td><td><a href='{{ url_for("report", patient_id=p.patient_id) }}'>Report →</a></td></tr>{% endfor %}</table></div>{% else %}<p class='muted'>No assessments yet. Start with a synthetic patient assessment.</p>{% endif %}</div><div class='notice'>Synthetic-data/non-diagnostic demo: predictions are educational fallbacks, not medical advice or a clinical decision.</div>""", hospital=hospital, branch=BRANCHES[hospital], username=session.get("username"), store=store, patients=store.list_patients(5))

    @app.route("/predict", methods=["GET", "POST"])
    def predict() -> str | Response:
        """Render and process a branch-local patient risk assessment."""
        denied = require_login()
        if denied: return denied
        values: Dict[str, Any] = request.form.to_dict() if request.method == "POST" else {}
        if request.method == "POST":
            patient_id = values.get("patient_id", "").strip() or f"DEMO-{secrets.token_hex(4).upper()}"
            values["patient_id"] = patient_id
            if not values.get("age"):
                flash("Age is required for a meaningful demo assessment.")
            else:
                store = stores[current_hospital()]
                # If the ID already exists just go straight to its report
                if store.get_patient(patient_id) is not None:
                    return redirect(url_for("report", patient_id=patient_id))
                probability = _trained_probability(values, artifact)
                model_source = "HeartShield Federated Model (LogisticRegression · Trained across 5 Hospital Partitions)"
                if probability is None:
                    probability = _probability(values)
                    model_source = "Clinical heuristic demo fallback"
                values["model_source"] = model_source
                try:
                    store.insert_patient(patient_id, values)
                except ValueError:
                    # race / double-submit — the row is already there, go to report
                    return redirect(url_for("report", patient_id=patient_id))
                store.set_prediction(patient_id, probability)
                # ── X-ray inference (optional) ───────────────────────────────
                xray_file = request.files.get("xray_image")
                if xray_file and xray_file.filename and _XRAY_AVAILABLE:
                    try:
                        image_bytes = xray_file.read()
                        xray_preds = predict_xray_from_bytes(current_hospital(), image_bytes)
                        mime = xray_file.content_type or "image/png"
                        img_b64 = f"data:{mime};base64," + base64.b64encode(image_bytes).decode("ascii")
                        store.set_xray_predictions(patient_id, xray_preds, image_data=img_b64)
                    except Exception as err:
                        print(f"X-ray prediction error: {err}")
                return redirect(url_for("report", patient_id=patient_id))
        return _page("New assessment", """<div class='row'><div><div class='eyebrow'>Branch {{ hospital }} · New assessment</div><h2>Clinical risk intake</h2><p class='muted'>Enter synthetic or de-identified values. Required fields are marked.</p></div></div><div class='notice'>This tool is a research demonstration. It does not diagnose, triage, or replace professional clinical judgment.</div><div style='background:#f0f7ff;border:1px solid #bfdbfe;border-radius:12px;padding:14px 18px;margin-bottom:18px'><div style='font-size:12px;font-weight:700;color:#1e40af;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px'>⚡ Quick Fill Demo Profiles</div><div style='display:flex;gap:10px;flex-wrap:wrap'><button type='button' class='btn secondary' onclick='fillPreset("high")' style='font-size:12px;padding:6px 12px'>🔴 High Risk Patient (68y, HTN, Angina)</button><button type='button' class='btn secondary' onclick='fillPreset("mod")' style='font-size:12px;padding:6px 12px'>🟡 Moderate Risk (52y, Borderline BP)</button><button type='button' class='btn secondary' onclick='fillPreset("low")' style='font-size:12px;padding:6px 12px'>🟢 Low Risk (29y, Normal Labs)</button></div></div><form method='post' enctype='multipart/form-data' class='card'><div class='form-grid'><div class='field'><label>Patient ID <span class='muted'>(optional)</span></label><input name='patient_id' value='{{ values.get("patient_id", "") }}' placeholder='Auto-generated if blank'></div>{% for key,label,kind,unit in fields %}<div class='field'><label>{{ label }} <span class='muted'>{{ unit }}</span>{% if key == "age" %} *{% endif %}</label>{% if kind == "select" %}<select name='{{ key }}'><option value=''>Unknown</option>{% for val,text in selects.get(key, []) %}<option value='{{ val }}' {% if values.get(key)==val %}selected{% endif %}>{{ text }}</option>{% endfor %}</select>{% else %}<input type='number' step='any' name='{{ key }}' value='{{ values.get(key, "") }}' {% if key == "age" %}required{% endif %}>{% endif %}</div>{% endfor %}</div><div style='margin-top:24px'><div class='eyebrow' style='margin-bottom:12px'>Chest X-ray (optional)</div><label class='xray-upload' id='xray-label'><input type='file' name='xray_image' id='xray-input' accept='image/*,.dcm'><svg width='36' height='36' fill='none' stroke='#94a3b8' stroke-width='1.5' viewBox='0 0 24 24'><path d='M12 16V8m0 0-3 3m3-3 3 3'/><rect x='3' y='3' width='18' height='18' rx='3'/></svg><b>Upload chest X-ray</b><small>PNG, JPG accepted · optional</small><img id='xray-preview' style='display:none;max-width:180px;max-height:180px;margin:12px auto 0;border-radius:8px;border:1px solid var(--line);object-fit:contain;' alt='Preview'>{% if not xray_available %}<span style='display:block;margin-top:6px;font-size:11px;color:#805b00'>ℹ X-ray AI model not loaded — install torch & torchvision to enable automated analysis</span>{% endif %}</label><div class='xray-filename' id='xray-name' style='display:none'></div></div><div class='actions'><button class='btn'>Generate local prediction</button><a class='btn secondary' href='{{ url_for("dashboard") }}'>Cancel</a></div></form><script>function fillPreset(t){var p={high:{age:68,sex:'Male',bmi:31.2,heart_rate:88,systolic_bp:165,diastolic_bp:95,oxygen_saturation:94,diabetes:'1',hypertension:'1',coronary_artery_disease:'1',smoking_status:'current',family_history_cardiac_disease:'1',physical_activity_level:'low',chest_pain:'1',shortness_of_breath:'1',fatigue:'1',palpitations:'1',total_cholesterol:250,ldl_cholesterol:160,hdl_cholesterol:36,triglycerides:215,blood_glucose:165,hba1c:7.9,creatinine:1.4},mod:{age:52,sex:'Female',bmi:27.4,heart_rate:76,systolic_bp:138,diastolic_bp:85,oxygen_saturation:97,diabetes:'0',hypertension:'1',coronary_artery_disease:'0',smoking_status:'former',physical_activity_level:'moderate',family_history_cardiac_disease:'1',chest_pain:'0',shortness_of_breath:'0',fatigue:'0',palpitations:'0',total_cholesterol:205,ldl_cholesterol:120,hdl_cholesterol:48,triglycerides:155,blood_glucose:108,hba1c:5.8,creatinine:0.95},low:{age:29,sex:'Female',bmi:21.8,heart_rate:68,systolic_bp:115,diastolic_bp:74,oxygen_saturation:99,diabetes:'0',hypertension:'0',coronary_artery_disease:'0',smoking_status:'never',physical_activity_level:'high',family_history_cardiac_disease:'0',chest_pain:'0',shortness_of_breath:'0',fatigue:'0',palpitations:'0',total_cholesterol:165,ldl_cholesterol:92,hdl_cholesterol:58,triglycerides:105,blood_glucose:88,hba1c:5.0,creatinine:0.8}}[t];if(!p)return;for(var k in p){var el=document.querySelector('[name=\"'+k+'\"]');if(el)el.value=p[k];}}var inp=document.getElementById('xray-input');if(inp){inp.addEventListener('change',function(){var lbl=document.getElementById('xray-label');var nm=document.getElementById('xray-name');var prev=document.getElementById('xray-preview');if(inp.files.length){lbl.style.borderColor='var(--blue)';nm.textContent='Selected: '+inp.files[0].name;nm.style.display='block';if(inp.files[0].type.startsWith('image/')){var r=new FileReader();r.onload=function(e){if(prev){prev.src=e.target.result;prev.style.display='block';}};r.readAsDataURL(inp.files[0]);}}});}</script>""",
        hospital=current_hospital(), fields=FIELDS, selects=SELECTS, values=values, xray_available=_XRAY_AVAILABLE)

    @app.get("/patients")
    def patients() -> str | Response:
        """Render only the authenticated branch's local patient index."""
        denied = require_login()
        if denied: return denied
        hospital = current_hospital()
        return _page("Patients", """<div class='row'><div><div class='eyebrow'>{{ branch.name }}</div><h2>Local assessments</h2><p class='muted'>Only records from branch {{ hospital }} are visible.</p></div><a class='btn' href='{{ url_for("predict") }}'>＋ New assessment</a></div><div class='card' style='margin-top:22px'><div class='table-wrap'><table class='table'><tr><th>Patient ID</th><th>Created</th><th>Prediction</th><th>Action</th></tr>{% for p in items %}<tr><td><b>{{ p.patient_id }}</b></td><td>{{ p.created_at }}</td><td><span class='pill {% if p.prediction and p.prediction >= 0.5 %}high{% endif %}'>{{ "%.0f"|format((p.prediction or 0)*100) }}% · {% if p.prediction and p.prediction >= 0.5 %}elevated{% else %}lower{% endif %}</span></td><td><a href='{{ url_for("report", patient_id=p.patient_id) }}'>Open report →</a></td></tr>{% else %}<tr><td colspan='4' class='muted'>No local assessments yet.</td></tr>{% endfor %}</table></div></div>""", hospital=hospital, branch=BRANCHES[hospital], items=stores[hospital].list_patients(100))

    @app.get("/report/<patient_id>")
    def report(patient_id: str) -> str | Response:
        """Render a downloadable HTML report for one local patient."""
        denied = require_login()
        if denied: return denied
        patient = stores[current_hospital()].get_patient(patient_id)
        if patient is None: return _page("Not found", "<div class='card'><h2>Assessment not found</h2><p class='muted'>This ID is not present in your branch.</p></div>"), 404
        probability = float(patient.get("prediction") or _probability(patient))
        risk = "Elevated" if probability >= 0.5 else "Lower"
        factors = _factors(patient, artifact)
        model_source = str(patient.get("model_source", "HeartShield Federated Model"))
        xray_preds: Optional[Dict[str, float]] = patient.get("xray_predictions")
        xray_image: Optional[str] = patient.get("xray_image")

        # Sort findings by probability descending for display
        xray_sorted = sorted(xray_preds.items(), key=lambda x: x[1], reverse=True) if xray_preds else []

        # Cardiomegaly gets a highlight if present and notable (>= 0.3)
        cardiomegaly_prob = xray_preds.get("Cardiomegaly", 0.0) if xray_preds else 0.0

        body = """<div class='row'><div><div class='eyebrow'>HeartShield Network · Branch {{ hospital }}</div><h2>Assessment report</h2><p class='muted'>Patient ID: <b>{{ patient_id }}</b> · {{ generated }}</p></div><button class='btn secondary' onclick='window.print()'>Download / print</button></div>
<div class='grid' style='margin-top:22px'>
  <div class='card'><div class='stat'>{{ "%.0f"|format(probability*100) }}%</div><div class='stat-label'>Cardiac risk score</div></div>
  <div class='card'><div class='stat' style='color:{% if probability >= 0.5 %}#c5564c{% else %}#119c88{% endif %}'>{{ risk }}</div><div class='stat-label'>Risk band</div></div>
  <div class='card'><div class='stat'>Local</div><div class='stat-label'>Data residency</div></div>
</div>
<div class='card' style='margin-top:22px'>
  <h3 class='section-title'>Tabular model</h3>
  <p style='margin:0 0 14px;color:var(--muted);font-size:13px'>{{ model_source }}</p>
  <h3 class='section-title'>Important contributing factors</h3>
  <div class='factors'>{% for name,value in factors %}<div class='factor'><span>{{ name }}</span><div class='bar'><i style='width:{{ value*100 if value*100 < 100 else 100 }}%'></i></div><b>{{ "%.0f"|format(value*100) }}</b></div>{% endfor %}</div>
  <p class='fine' style='margin-top:16px'>Factors are model-derived contributions indicating clinical impact on the risk score.</p>
</div>
{% if xray_sorted %}
<div class='card xray-section'>
  <div class='row' style='margin-bottom:16px'>
    <div>
      <h3 class='section-title' style='margin:0'>Chest X-ray AI Analysis</h3>
      <p class='fine' style='margin:4px 0 0'>Federated ResNet18 · Multi-Label NIH ChestX-ray14 Classifier</p>
    </div>
    <span class='pill'>Hospital {{ hospital }} Local Model</span>
  </div>
  {% if xray_image %}
  <div style='display:flex;gap:22px;margin-bottom:18px;align-items:flex-start;flex-wrap:wrap;'>
    <div style='text-align:center;'>
      <img src='{{ xray_image }}' alt='Uploaded Chest X-ray' style='max-width:220px;max-height:220px;border-radius:10px;border:1px solid var(--line);box-shadow:0 3px 12px rgba(0,0,0,0.1);object-fit:contain;background:#10233f;display:block;'>
      <div style='font-size:11px;color:var(--muted);margin-top:6px;font-weight:600'>Uploaded X-Ray Scan</div>
    </div>
    <div style='flex:1;min-width:240px;'>
      {% if cardiomegaly_prob >= 0.3 %}
      <div class='xray-highlight' style='margin-bottom:0'>
        <div style='display:flex;align-items:center;gap:10px'>
          <svg width='22' height='22' fill='none' stroke='#0e7a6e' stroke-width='2' viewBox='0 0 24 24'><path d='M12 2C6.477 2 2 6.477 2 12s4.477 10 10 10 10-4.477 10-10S17.523 2 12 2zm0 5v4m0 4h.01'/></svg>
          <div>
            <b style='color:#0e5c55'>Cardiomegaly</b>
            <span style='color:#0e7a6e;font-size:13px'> · {{ "%.0f"|format(cardiomegaly_prob*100) }}% probability</span>
            {% if cardiomegaly_prob >= 0.5 %}<span class='badge-high'>Notable</span>{% else %}<span class='badge-mod'>Borderline</span>{% endif %}
            <p style='margin:4px 0 0;font-size:12px;color:#0e7a6e'>Enlarged heart detected on X-ray — corroborates cardiac risk assessment.</p>
          </div>
        </div>
      </div>
      {% else %}
      <p style='margin:0;color:var(--muted);font-size:13px'>AI multi-label classification completed for 14 chest conditions. Results reflect per-hospital federated training weights.</p>
      {% endif %}
    </div>
  </div>
  {% elif cardiomegaly_prob >= 0.3 %}
  <div class='xray-highlight'>
    <div style='display:flex;align-items:center;gap:10px'>
      <svg width='22' height='22' fill='none' stroke='#0e7a6e' stroke-width='2' viewBox='0 0 24 24'><path d='M12 2C6.477 2 2 6.477 2 12s4.477 10 10 10 10-4.477 10-10S17.523 2 12 2zm0 5v4m0 4h.01'/></svg>
      <div>
        <b style='color:#0e5c55'>Cardiomegaly</b>
        <span style='color:#0e7a6e;font-size:13px'> · {{ "%.0f"|format(cardiomegaly_prob*100) }}% probability</span>
        {% if cardiomegaly_prob >= 0.5 %}<span class='badge-high'>Notable</span>{% else %}<span class='badge-mod'>Borderline</span>{% endif %}
        <p style='margin:4px 0 0;font-size:12px;color:#0e7a6e'>Enlarged heart detected on X-ray — corroborates cardiac risk assessment above.</p>
      </div>
    </div>
  </div>
  {% endif %}
  <div style='display:grid;grid-template-columns:1fr 1fr;gap:6px 24px'>
  {% for label, prob in xray_sorted %}
  <div class='finding-row'>
    <span class='finding-label'>{{ label.replace("_"," ") }}</span>
    <div class='finding-bar'><div class='finding-fill' style='width:{{ (prob*100)|round|int }}%;background:{% if prob >= 0.5 %}#e05252{% elif prob >= 0.3 %}#f0bb4c{% else %}#19b99c{% endif %}'></div></div>
    <span class='finding-pct' style='color:{% if prob >= 0.5 %}#c5564c{% elif prob >= 0.3 %}#805b00{% else %}#0e7a6e{% endif %}'>{{ "%.0f"|format(prob*100) }}%</span>
  </div>
  {% endfor %}
  </div>
  <p class='fine' style='margin-top:16px'>Multi-label NIH ChestX-ray14 classifier. Federated training across 5 hospital partitions. Non-diagnostic research demonstration.</p>
</div>
{% else %}
<div class='card' style='margin-top:22px;border:1px dashed #cbd5e1;background:#fafcff'>
  <div style='display:flex;align-items:center;gap:14px'>
    <svg width='32' height='32' fill='none' stroke='#94a3b8' stroke-width='1.5' viewBox='0 0 24 24'><rect x='3' y='3' width='18' height='18' rx='3'/><path d='M12 16V8m0 0-3 3m3-3 3 3'/></svg>
    <div>
      <b style='color:#334155;font-size:14px'>No chest X-ray scan attached to this patient assessment</b>
      <p style='margin:3px 0 0;font-size:12px;color:var(--muted)'>This assessment evaluated tabular clinical biomarkers. To include deep learning analysis for 14 chest conditions (Cardiomegaly, Pneumonia, etc.), attach an X-ray when creating a new assessment.</p>
    </div>
  </div>
</div>
{% endif %}
<div class='notice'>Non-diagnostic synthetic demonstration. Do not use this report for patient care, treatment, or emergency decisions. Consult a qualified clinician.</div>"""
        return _page("Assessment report", body,
            hospital=current_hospital(), patient_id=patient_id, patient=patient,
            probability=probability, risk=risk, factors=factors,
            model_source=model_source, xray_sorted=xray_sorted,
            cardiomegaly_prob=cardiomegaly_prob, xray_image=xray_image,
            generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

    @app.get("/api/health")
    def health() -> Any:
        """Return service health without exposing patient records."""
        return jsonify({"status": "ok", "privacy": "hospital-local storage", "network": NETWORK_NAME})

    @app.get("/api/hospitals")
    def hospitals() -> Any:
        """List configured branches without patient data."""
        return jsonify({"hospitals": list(stores)})

    @app.post("/api/patients/<hospital>")
    def add_patient(hospital: str) -> Any:
        """Preserve the legacy local insertion API with duplicate validation."""
        if hospital not in stores: return jsonify({"error": "unknown hospital"}), 404
        payload: Dict[str, Any] = request.get_json(silent=True) or {}; patient_id = str(payload.pop("patient_id", "")).strip()
        try:
            if stores[hospital].get_patient(patient_id) is not None: return jsonify({"error": "patient_id already exists"}), 409
            stores[hospital].insert_patient(patient_id, payload)
        except (ValueError, TypeError) as error: return jsonify({"error": str(error)}), 400
        return jsonify({"status": "stored", "hospital": hospital, "patient_id": patient_id}), 201

    return app