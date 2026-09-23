# ============================================================
# CELL 1: Install dependencies (run this cell first)
# ============================================================
# !pip install scikit-learn pandas numpy

# ============================================================
# CELL 2: Upload your hospital CSV files
# ============================================================
# Option A: Upload from local machine
from google.colab import files
uploaded = files.upload()  # Upload all hospital CSV files here

# Option B (alternative): Mount Google Drive instead
# from google.colab import drive
# drive.mount('/content/drive')
# DATA_DIR = '/content/drive/MyDrive/your_folder/'  # change this path

# ============================================================
# CELL 3: Imports and Setup
# ============================================================
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
import warnings
warnings.filterwarnings('ignore')

# Configuration
TARGET = "cardiac_risk_label"
HOSPITALS = ["A", "B", "C", "D", "E"]
MAX_ITER = 200
RANDOM_SEED = 42

# Categorical feature definitions
CATEGORICAL_VALUES = {
    "sex": ["Female", "Male", "Other", "Unknown"],
    "smoking_status": ["never", "former", "current", "Unknown"],
    "physical_activity_level": ["low", "moderate", "high", "Unknown"],
}
EXCLUDED_COLS = {"patient_id", "hospital_id", "cardiac_risk_label"}

# ============================================================
# CELL 4: Preprocessing functions (replaces fedmedtrust package)
# ============================================================
def build_feature_schema(columns):
    """Build numeric and categorical feature lists from column names."""
    numeric = []
    categorical = {}
    for col in columns:
        if col in EXCLUDED_COLS:
            continue
        if col in CATEGORICAL_VALUES:
            categorical[col] = CATEGORICAL_VALUES[col]
        else:
            numeric.append(col)
    return numeric, categorical

def transform(df, numeric_cols, categorical_cols):
    """Transform a DataFrame into a numeric feature matrix."""
    blocks = []
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        blocks.append(values.reshape(-1, 1))
    for col, categories in categorical_cols.items():
        values = df[col].fillna("Unknown").astype(str).to_numpy()
        blocks.append(np.column_stack([(values == cat).astype(float) for cat in categories]))
    return np.hstack(blocks) if blocks else np.empty((len(df), 0), dtype=float)

def evaluate(y_true, y_prob, threshold=0.5):
    """Compute classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if np.unique(y_true).size > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = 0.5
        metrics["pr_auc"] = 0.0
    return metrics

# Build schema from first hospital file
ref_df = pd.read_csv("hospital_A_train.csv", nrows=0)
numeric_cols, categorical_cols = build_feature_schema(ref_df.columns)
print(f"Numeric features: {len(numeric_cols)}")
print(f"Categorical features: {len(categorical_cols)}")
print("Setup complete!")

# ============================================================
# CELL 5: Train Hospital A
# ============================================================
hospital = "A"
print(f"\n{'='*60}")
print(f"  HOSPITAL {hospital} - Logistic Regression")
print(f"{'='*60}")

train_df = pd.read_csv(f"hospital_{hospital}_train.csv")
val_df = pd.read_csv(f"hospital_{hospital}_validation.csv")
test_df = pd.read_csv(f"hospital_{hospital}_test.csv")

X_train = transform(train_df, numeric_cols, categorical_cols)
y_train = train_df[TARGET].astype(int).to_numpy()
X_val = transform(val_df, numeric_cols, categorical_cols)
y_val = val_df[TARGET].astype(int).to_numpy()
X_test = transform(test_df, numeric_cols, categorical_cols)
y_test = test_df[TARGET].astype(int).to_numpy()

print(f"  Train: {len(y_train)} samples (pos={y_train.sum()}, neg={len(y_train)-y_train.sum()})")
print(f"  Val:   {len(y_val)} samples")
print(f"  Test:  {len(y_test)} samples")

model_A = LogisticRegression(max_iter=MAX_ITER, class_weight="balanced", random_state=RANDOM_SEED, solver="lbfgs")
model_A.fit(X_train, y_train)

train_metrics = evaluate(y_train, model_A.predict_proba(X_train)[:, 1])
val_metrics = evaluate(y_val, model_A.predict_proba(X_val)[:, 1])
test_metrics = evaluate(y_test, model_A.predict_proba(X_test)[:, 1])

print(f"\n  Train Metrics:      {train_metrics}")
print(f"  Validation Metrics: {val_metrics}")
print(f"  Test Metrics:       {test_metrics}")

y_pred = (model_A.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
print(f"\n  Classification Report (Test):")
print(classification_report(y_test, y_pred, target_names=["No Risk", "Risk"]))
print(f"  Confusion Matrix (Test):")
print(confusion_matrix(y_test, y_pred))

# ============================================================
# CELL 6: Train Hospital B
# ============================================================
hospital = "B"
print(f"\n{'='*60}")
print(f"  HOSPITAL {hospital} - Logistic Regression")
print(f"{'='*60}")

train_df = pd.read_csv(f"hospital_{hospital}_train.csv")
val_df = pd.read_csv(f"hospital_{hospital}_validation.csv")
test_df = pd.read_csv(f"hospital_{hospital}_test.csv")

X_train = transform(train_df, numeric_cols, categorical_cols)
y_train = train_df[TARGET].astype(int).to_numpy()
X_val = transform(val_df, numeric_cols, categorical_cols)
y_val = val_df[TARGET].astype(int).to_numpy()
X_test = transform(test_df, numeric_cols, categorical_cols)
y_test = test_df[TARGET].astype(int).to_numpy()

print(f"  Train: {len(y_train)} samples (pos={y_train.sum()}, neg={len(y_train)-y_train.sum()})")
print(f"  Val:   {len(y_val)} samples")
print(f"  Test:  {len(y_test)} samples")

model_B = LogisticRegression(max_iter=MAX_ITER, class_weight="balanced", random_state=RANDOM_SEED, solver="lbfgs")
model_B.fit(X_train, y_train)

train_metrics = evaluate(y_train, model_B.predict_proba(X_train)[:, 1])
val_metrics = evaluate(y_val, model_B.predict_proba(X_val)[:, 1])
test_metrics = evaluate(y_test, model_B.predict_proba(X_test)[:, 1])

print(f"\n  Train Metrics:      {train_metrics}")
print(f"  Validation Metrics: {val_metrics}")
print(f"  Test Metrics:       {test_metrics}")

y_pred = (model_B.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
print(f"\n  Classification Report (Test):")
print(classification_report(y_test, y_pred, target_names=["No Risk", "Risk"]))
print(f"  Confusion Matrix (Test):")
print(confusion_matrix(y_test, y_pred))

# ============================================================
# CELL 7: Train Hospital C
# ============================================================
hospital = "C"
print(f"\n{'='*60}")
print(f"  HOSPITAL {hospital} - Logistic Regression")
print(f"{'='*60}")

train_df = pd.read_csv(f"hospital_{hospital}_train.csv")
val_df = pd.read_csv(f"hospital_{hospital}_validation.csv")
test_df = pd.read_csv(f"hospital_{hospital}_test.csv")

X_train = transform(train_df, numeric_cols, categorical_cols)
y_train = train_df[TARGET].astype(int).to_numpy()
X_val = transform(val_df, numeric_cols, categorical_cols)
y_val = val_df[TARGET].astype(int).to_numpy()
X_test = transform(test_df, numeric_cols, categorical_cols)
y_test = test_df[TARGET].astype(int).to_numpy()

print(f"  Train: {len(y_train)} samples (pos={y_train.sum()}, neg={len(y_train)-y_train.sum()})")
print(f"  Val:   {len(y_val)} samples")
print(f"  Test:  {len(y_test)} samples")

model_C = LogisticRegression(max_iter=MAX_ITER, class_weight="balanced", random_state=RANDOM_SEED, solver="lbfgs")
model_C.fit(X_train, y_train)

train_metrics = evaluate(y_train, model_C.predict_proba(X_train)[:, 1])
val_metrics = evaluate(y_val, model_C.predict_proba(X_val)[:, 1])
test_metrics = evaluate(y_test, model_C.predict_proba(X_test)[:, 1])

print(f"\n  Train Metrics:      {train_metrics}")
print(f"  Validation Metrics: {val_metrics}")
print(f"  Test Metrics:       {test_metrics}")

y_pred = (model_C.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
print(f"\n  Classification Report (Test):")
print(classification_report(y_test, y_pred, target_names=["No Risk", "Risk"]))
print(f"  Confusion Matrix (Test):")
print(confusion_matrix(y_test, y_pred))

# ============================================================
# CELL 8: Train Hospital D
# ============================================================
hospital = "D"
print(f"\n{'='*60}")
print(f"  HOSPITAL {hospital} - Logistic Regression")
print(f"{'='*60}")

train_df = pd.read_csv(f"hospital_{hospital}_train.csv")
val_df = pd.read_csv(f"hospital_{hospital}_validation.csv")
test_df = pd.read_csv(f"hospital_{hospital}_test.csv")

X_train = transform(train_df, numeric_cols, categorical_cols)
y_train = train_df[TARGET].astype(int).to_numpy()
X_val = transform(val_df, numeric_cols, categorical_cols)
y_val = val_df[TARGET].astype(int).to_numpy()
X_test = transform(test_df, numeric_cols, categorical_cols)
y_test = test_df[TARGET].astype(int).to_numpy()

print(f"  Train: {len(y_train)} samples (pos={y_train.sum()}, neg={len(y_train)-y_train.sum()})")
print(f"  Val:   {len(y_val)} samples")
print(f"  Test:  {len(y_test)} samples")

model_D = LogisticRegression(max_iter=MAX_ITER, class_weight="balanced", random_state=RANDOM_SEED, solver="lbfgs")
model_D.fit(X_train, y_train)

train_metrics = evaluate(y_train, model_D.predict_proba(X_train)[:, 1])
val_metrics = evaluate(y_val, model_D.predict_proba(X_val)[:, 1])
test_metrics = evaluate(y_test, model_D.predict_proba(X_test)[:, 1])

print(f"\n  Train Metrics:      {train_metrics}")
print(f"  Validation Metrics: {val_metrics}")
print(f"  Test Metrics:       {test_metrics}")

y_pred = (model_D.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
print(f"\n  Classification Report (Test):")
print(classification_report(y_test, y_pred, target_names=["No Risk", "Risk"]))
print(f"  Confusion Matrix (Test):")
print(confusion_matrix(y_test, y_pred))

# ============================================================
# CELL 9: Train Hospital E
# ============================================================
hospital = "E"
print(f"\n{'='*60}")
print(f"  HOSPITAL {hospital} - Logistic Regression")
print(f"{'='*60}")

train_df = pd.read_csv(f"hospital_{hospital}_train.csv")
val_df = pd.read_csv(f"hospital_{hospital}_validation.csv")
test_df = pd.read_csv(f"hospital_{hospital}_test.csv")

X_train = transform(train_df, numeric_cols, categorical_cols)
y_train = train_df[TARGET].astype(int).to_numpy()
X_val = transform(val_df, numeric_cols, categorical_cols)
y_val = val_df[TARGET].astype(int).to_numpy()
X_test = transform(test_df, numeric_cols, categorical_cols)
y_test = test_df[TARGET].astype(int).to_numpy()

print(f"  Train: {len(y_train)} samples (pos={y_train.sum()}, neg={len(y_train)-y_train.sum()})")
print(f"  Val:   {len(y_val)} samples")
print(f"  Test:  {len(y_test)} samples")

model_E = LogisticRegression(max_iter=MAX_ITER, class_weight="balanced", random_state=RANDOM_SEED, solver="lbfgs")
model_E.fit(X_train, y_train)

train_metrics = evaluate(y_train, model_E.predict_proba(X_train)[:, 1])
val_metrics = evaluate(y_val, model_E.predict_proba(X_val)[:, 1])
test_metrics = evaluate(y_test, model_E.predict_proba(X_test)[:, 1])

print(f"\n  Train Metrics:      {train_metrics}")
print(f"  Validation Metrics: {val_metrics}")
print(f"  Test Metrics:       {test_metrics}")

y_pred = (model_E.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
print(f"\n  Classification Report (Test):")
print(classification_report(y_test, y_pred, target_names=["No Risk", "Risk"]))
print(f"  Confusion Matrix (Test):")
print(confusion_matrix(y_test, y_pred))

# ============================================================
# CELL 10: Summary comparison of all hospitals
# ============================================================
print(f"\n{'='*60}")
print(f"  SUMMARY - All Hospitals Comparison")
print(f"{'='*60}")

summary_rows = []
for name, model in [("A", model_A), ("B", model_B), ("C", model_C), ("D", model_D), ("E", model_E)]:
    test_df = pd.read_csv(f"hospital_{name}_test.csv")
    X_test = transform(test_df, numeric_cols, categorical_cols)
    y_test = test_df[TARGET].astype(int).to_numpy()
    test_prob = model.predict_proba(X_test)[:, 1]
    m = evaluate(y_test, test_prob)
    m["hospital"] = name
    summary_rows.append(m)

summary_df = pd.DataFrame(summary_rows)[["hospital", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]]
print(summary_df.to_string(index=False))
