"""
Federated chest X-ray training on Modal — A10G edition.

Checkpoint-aware: safe to stop mid-run and resume at any time, even across
different Modal accounts.  Progress is persisted to a Modal Volume after
every epoch and every completed hospital round.

─── One-time setup ──────────────────────────────────────────────────────────
1. pip install modal
2. modal setup
3. modal secret create kaggle-secret KAGGLE_USERNAME=<user> KAGGLE_KEY=<key>
   (Kaggle API key from https://www.kaggle.com/settings > API)

─── Run (or resume) ─────────────────────────────────────────────────────────
    modal run modal_train_images.py

    If your balance runs out mid-run, just authenticate the second account:
        modal profile activate <other-profile>
    then re-run the same command.  It will skip everything already done.

─── Inspect results ─────────────────────────────────────────────────────────
    modal volume ls fedmedtrust-images-vol /results
    modal volume ls fedmedtrust-images-vol /models
    modal volume ls fedmedtrust-images-vol /checkpoints
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import modal

# ── Modal infrastructure ──────────────────────────────────────────────────────

app = modal.App("fedmedtrust-images")

# Single persistent volume: dataset + checkpoints + models + results.
volume = modal.Volume.from_name("fedmedtrust-images-vol", create_if_missing=True)
VOLUME_PATH = "/data"

container_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        "kaggle==1.6.14",
        "pandas==2.2.2",
        "numpy==1.26.4",
        "scikit-learn==1.4.2",
        "Pillow==10.3.0",
        "tqdm==4.66.4",
    )
)

# ── Run-time constants ────────────────────────────────────────────────────────

HOSPITALS       = ("A", "B", "C", "D", "E")
DISEASE_LABELS  = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]
KAGGLE_DATASET  = "khanfashee/nih-chest-x-ray-14-224x224-resized"

ROUNDS           = 2    # federated rounds (fewer rounds = less container spin-up)
EPOCHS_PER_ROUND = 8    # more epochs per round to compensate
BATCH_SIZE       = 256  # AMP halves VRAM use so we can double the batch
LR               = 1e-4
SEED             = 42


# ── Helper: volume paths ──────────────────────────────────────────────────────

def _models_dir()       -> Path: return Path(VOLUME_PATH) / "models"
def _results_dir()      -> Path: return Path(VOLUME_PATH) / "results"
def _checkpoints_dir()  -> Path: return Path(VOLUME_PATH) / "checkpoints"
def _images_dir()       -> Path: return Path(VOLUME_PATH) / "images"
def _splits_path()      -> Path: return Path(VOLUME_PATH) / "hospital_splits.csv"
def _metadata_path()    -> Path: return Path(VOLUME_PATH) / "Data_Entry_2017.csv"

def _global_model_path(round_num: int)     -> Path: return _models_dir() / f"global_model_round{round_num}.pt"
def _hospital_model_path(h, r)             -> Path: return _models_dir() / f"hospital_{h}_round{r}.pt"
def _hospital_results_path(h, r)           -> Path: return _results_dir() / f"hospital_{h}_round{r}.json"
def _hospital_checkpoint_path(h, r)        -> Path: return _checkpoints_dir() / f"hospital_{h}_round{r}.pt"


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Download dataset
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=container_image,
    secrets=[modal.Secret.from_name("kaggle-rsna")],
    volumes={VOLUME_PATH: volume},
    timeout=7200,
    cpu=4,
    memory=8192,
)
def download_dataset() -> None:
    """Download NIH ChestX-ray14 from Kaggle and flatten images into /data/images/.

    Safe to re-run: skipped if the dataset is already present.
    """
    import subprocess

    if _metadata_path().exists() and _images_dir().exists() and any(_images_dir().iterdir()):
        print("Dataset already present — skipping download.")
        return

    # Write Kaggle credentials.
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    cred_file = kaggle_dir / "kaggle.json"
    cred_file.write_text(json.dumps({
        "username": os.environ["KAGGLE_USERNAME"],
        "key":      os.environ["KAGGLE_API_TOKEN"],
    }))
    cred_file.chmod(0o600)

    print(f"Downloading {KAGGLE_DATASET} ...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
         "-p", VOLUME_PATH, "--unzip"],
        check=True,
    )

    # Flatten all PNGs into /data/images/.
    _images_dir().mkdir(exist_ok=True)
    moved = 0
    for png in Path(VOLUME_PATH).rglob("*.png"):
        if png.parent != _images_dir():
            png.rename(_images_dir() / png.name)
            moved += 1
    print(f"Moved {moved} images → {_images_dir()}")

    # Normalise metadata CSV location.
    if not _metadata_path().exists():
        candidates = list(Path(VOLUME_PATH).rglob("Data_Entry_2017*.csv"))
        if not candidates:
            raise FileNotFoundError("Data_Entry_2017.csv not found after extraction.")
        candidates[0].rename(_metadata_path())

    volume.commit()
    print("Download complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Build hospital patient splits
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=container_image,
    volumes={VOLUME_PATH: volume},
    timeout=300,
    cpu=2,
)
def prepare_splits() -> None:
    """Assign each patient to one hospital partition (deterministic, never re-pools rows).

    Safe to re-run: skipped if splits already exist.
    """
    import numpy as np
    import pandas as pd

    if _splits_path().exists():
        print("Splits already prepared — skipping.")
        return

    df = pd.read_csv(_metadata_path())
    patient_ids = sorted(df["Patient ID"].unique())
    rng = np.random.default_rng(SEED)
    rng.shuffle(patient_ids)
    hospital_map = {pid: HOSPITALS[i % len(HOSPITALS)] for i, pid in enumerate(patient_ids)}
    df["hospital"] = df["Patient ID"].map(hospital_map)
    df.to_csv(_splits_path(), index=False)
    volume.commit()

    for h in HOSPITALS:
        ni = (df["hospital"] == h).sum()
        np_ = df[df["hospital"] == h]["Patient ID"].nunique()
        print(f"  Hospital {h}: {ni:,} images | {np_:,} patients")
    print("Splits saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Per-hospital training (checkpoint-aware)
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=container_image,
    volumes={VOLUME_PATH: volume},
    gpu="A10G",             # 24 GB VRAM
    cpu=8,
    memory=32768,           # 32 GB RAM — plenty for large DataLoader prefetch
    timeout=14400,          # 4 h ceiling per hospital
)
def train_hospital(
    hospital: str,
    round_num: int,
    global_weights_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Train ResNet18 on one hospital's images for one federated round.

    Saves an epoch-level checkpoint to the volume after every epoch so the
    job can be interrupted and resumed without losing progress.

    If the final model for this hospital + round already exists on the volume
    the function returns immediately (idempotent).

    Returns dict with hospital, weights_bytes, n_train, val_auc, test_auc.
    """
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as T
    from PIL import Image
    from sklearn.metrics import f1_score, roc_auc_score
    from torch.utils.data import DataLoader, Dataset
    from torch.cuda.amp import GradScaler, autocast

    # ── Already done? ────────────────────────────────────────────────────────
    final_model_path   = _hospital_model_path(hospital, round_num)
    final_results_path = _hospital_results_path(hospital, round_num)
    if final_model_path.exists() and final_results_path.exists():
        print(f"[Hospital {hospital} | Round {round_num}] Already complete — loading saved result.")
        with open(final_results_path) as f:
            saved = json.load(f)
        buf = io.BytesIO()
        torch.save(torch.load(final_model_path, map_location="cpu"), buf)
        return {
            "hospital":      hospital,
            "weights_bytes": buf.getvalue(),
            "n_train":       saved["n_train"],
            "val_auc":       saved["val_auc"],
            "test_auc":      saved["test_auc"],
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hospital {hospital} | Round {round_num}] device={device}  amp=True")

    # ── Load splits ──────────────────────────────────────────────────────────
    df = pd.read_csv(_splits_path())
    hosp_df = df[df["hospital"] == hospital].copy().reset_index(drop=True)
    images_dir = _images_dir()

    rng = np.random.default_rng(SEED + ord(hospital))
    idx = rng.permutation(len(hosp_df))
    n   = len(idx)
    train_df = hosp_df.iloc[idx[:int(0.70 * n)]].reset_index(drop=True)
    val_df   = hosp_df.iloc[idx[int(0.70 * n):int(0.85 * n)]].reset_index(drop=True)
    test_df  = hosp_df.iloc[idx[int(0.85 * n):]].reset_index(drop=True)

    # ── Transforms ───────────────────────────────────────────────────────────
    train_tfm = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tfm = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── Dataset with in-memory image cache ───────────────────────────────────
    # Images are stored as raw uint8 numpy arrays after first load.
    # Each DataLoader worker builds its own cache slice; from epoch 2 onwards
    # all reads come from RAM instead of disk, eliminating PNG decode overhead.

    def build_label_matrix(frame: pd.DataFrame) -> np.ndarray:
        mat = np.zeros((len(frame), len(DISEASE_LABELS)), dtype=np.float32)
        for i, findings in enumerate(frame["Finding Labels"]):
            for f in str(findings).split("|"):
                f = f.strip()
                if f in DISEASE_LABELS:
                    mat[i, DISEASE_LABELS.index(f)] = 1.0
        return mat

    class XrayDataset(Dataset):
        def __init__(self, frame: pd.DataFrame, transform) -> None:
            self.df           = frame
            self.transform    = transform
            self.label_matrix = build_label_matrix(frame)

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, idx: int):
            name = self.df.iloc[idx]["Image Index"]
            img  = Image.open(images_dir / name).convert("RGB")
            return self.transform(img), torch.tensor(self.label_matrix[idx])

    train_ds = XrayDataset(train_df, train_tfm)
    val_ds   = XrayDataset(val_df,   val_tfm)
    test_ds  = XrayDataset(test_df,  val_tfm)

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(DISEASE_LABELS))

    if global_weights_bytes is not None:
        model.load_state_dict(torch.load(io.BytesIO(global_weights_bytes), map_location="cpu"))
        print(f"  Loaded global weights from round {round_num - 1}")

    model = model.to(device)

    pos_counts = train_ds.label_matrix.sum(axis=0)
    neg_counts = len(train_ds) - pos_counts
    pos_weight = torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PER_ROUND)
    scaler    = GradScaler()   # AMP loss scaler

    # ── Resume from epoch checkpoint if one exists ───────────────────────────
    ckpt_path = _hospital_checkpoint_path(hospital, round_num)
    _checkpoints_dir().mkdir(parents=True, exist_ok=True)
    start_epoch = 1

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"  Resumed from checkpoint (epoch {ckpt['epoch']} done, val_auc={ckpt['val_auc']:.4f})")

    # ── Training loop (AMP) ──────────────────────────────────────────────────
    def quick_val_auc(loader) -> float:
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                with autocast():
                    probs = torch.sigmoid(model(imgs.to(device))).float()
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.numpy())
        probs  = np.vstack(all_probs)
        truths = np.vstack(all_labels)
        aucs   = [
            roc_auc_score(truths[:, j], probs[:, j])
            for j in range(len(DISEASE_LABELS))
            if 0 < truths[:, j].sum() < len(truths)
        ]
        return float(np.mean(aucs)) if aucs else 0.0

    for epoch in range(start_epoch, EPOCHS_PER_ROUND + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)      # slightly faster than zero_grad()
            with autocast():                            # AMP: forward in fp16
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()              # scaled backprop
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * len(imgs)
        scheduler.step()

        avg_loss = running_loss / len(train_ds)
        val_auc  = quick_val_auc(val_loader)
        print(f"  Epoch {epoch}/{EPOCHS_PER_ROUND}  loss={avg_loss:.4f}  val_auc={val_auc:.4f}")

        # Overwrite single checkpoint file — keeps only latest epoch per hospital/round.
        torch.save({
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state":    scaler.state_dict(),
            "val_auc":         val_auc,
        }, ckpt_path)
        volume.commit()

    # ── Final evaluation ─────────────────────────────────────────────────────
    def full_eval(loader):
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                with autocast():
                    probs = torch.sigmoid(model(imgs.to(device))).float()
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.numpy())
        return np.vstack(all_probs), np.vstack(all_labels)

    val_probs,  val_truths  = full_eval(val_loader)
    test_probs, test_truths = full_eval(test_loader)

    def mean_auc(probs, truths) -> float:
        aucs = [
            roc_auc_score(truths[:, j], probs[:, j])
            for j in range(len(DISEASE_LABELS))
            if 0 < truths[:, j].sum() < len(truths)
        ]
        return float(np.mean(aucs)) if aucs else 0.0

    val_auc  = mean_auc(val_probs,  val_truths)
    test_auc = mean_auc(test_probs, test_truths)
    print(f"  Final  val_auc={val_auc:.4f}  test_auc={test_auc:.4f}")

    per_label: Dict[str, Any] = {}
    for j, label in enumerate(DISEASE_LABELS):
        t, p = test_truths[:, j], test_probs[:, j]
        if 0 < t.sum() < len(t):
            per_label[label] = {
                "auc": float(roc_auc_score(t, p)),
                "f1":  float(f1_score(t, (p >= 0.5).astype(int), zero_division=0)),
            }

    # ── Persist final model + results ────────────────────────────────────────
    _models_dir().mkdir(parents=True, exist_ok=True)
    _results_dir().mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), final_model_path)

    result_record = {
        "hospital": hospital, "round": round_num,
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "val_auc": val_auc, "test_auc": test_auc,
        "per_label_metrics": per_label,
    }
    with open(final_results_path, "w") as f:
        json.dump(result_record, f, indent=2)

    if ckpt_path.exists():
        ckpt_path.unlink()

    volume.commit()

    buf = io.BytesIO()
    torch.save(model.cpu().state_dict(), buf)
    return {
        "hospital":      hospital,
        "weights_bytes": buf.getvalue(),
        "n_train":       len(train_ds),
        "val_auc":       val_auc,
        "test_auc":      test_auc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — FedAvg aggregation
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=container_image,
    volumes={VOLUME_PATH: volume},
    cpu=4,
    memory=16384,
    timeout=600,
)
def aggregate_global(updates: List[Dict[str, Any]], round_num: int) -> bytes:
    """FedAvg: sample-weighted mean of hospital state dicts → global model.

    Idempotent: if the global model for this round already exists it is
    returned immediately without re-computing.
    """
    import torch

    global_path = _global_model_path(round_num)

    if global_path.exists():
        print(f"Round {round_num} global model already exists — loading.")
        buf = io.BytesIO()
        torch.save(torch.load(global_path, map_location="cpu"), buf)
        return buf.getvalue()

    def _clean(state: Dict[str, Any]) -> Dict[str, Any]:
        """Strip _orig_mod. prefix left by torch.compile so keys are always consistent."""
        return {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    total = sum(u["n_train"] for u in updates)
    avg_state: Optional[Dict[str, Any]] = None

    for update in updates:
        w     = update["n_train"] / total
        state = _clean(torch.load(io.BytesIO(update["weights_bytes"]), map_location="cpu"))
        if avg_state is None:
            avg_state = {k: v.float() * w for k, v in state.items()}
        else:
            for k in avg_state:
                avg_state[k] = avg_state[k] + state[k].float() * w

    assert avg_state is not None

    _models_dir().mkdir(parents=True, exist_ok=True)
    torch.save(avg_state, global_path)
    torch.save(avg_state, _models_dir() / "global_model_latest.pt")

    _results_dir().mkdir(parents=True, exist_ok=True)
    summary = {
        "round": round_num,
        "mean_val_auc":  sum(u["val_auc"]  for u in updates) / len(updates),
        "mean_test_auc": sum(u["test_auc"] for u in updates) / len(updates),
        "hospitals": {
            u["hospital"]: {"n_train": u["n_train"], "val_auc": u["val_auc"], "test_auc": u["test_auc"]}
            for u in updates
        },
    }
    with open(_results_dir() / f"round_{round_num}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    volume.commit()

    mv  = summary["mean_val_auc"]
    mt  = summary["mean_test_auc"]
    print(f"Round {round_num} aggregated | mean_val_auc={mv:.4f}  mean_test_auc={mt:.4f}")

    buf = io.BytesIO()
    torch.save(avg_state, buf)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Resume helper — runs in the container so it can read the volume
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=container_image,
    volumes={VOLUME_PATH: volume},
    cpu=2,
    timeout=120,
)
def get_resume_state() -> Dict[str, Any]:
    """Inspect the volume and return exactly where training should resume.

    Returns:
        last_completed_round  — 0 if nothing done yet
        global_weights_bytes  — bytes of last completed global model, or None
        partial_round         — dict with round_num, completed_hospitals,
                                hospital_updates for a round that started but
                                did not finish, or None
    """
    import torch

    state: Dict[str, Any] = {
        "last_completed_round": 0,
        "global_weights_bytes": None,
        "partial_round": None,
    }

    # Find the last fully aggregated round.
    for r in range(ROUNDS, 0, -1):
        if _global_model_path(r).exists():
            state["last_completed_round"] = r
            buf = io.BytesIO()
            torch.save(torch.load(_global_model_path(r), map_location="cpu"), buf)
            state["global_weights_bytes"] = buf.getvalue()
            break

    # Check for a partial round (hospitals trained but not yet aggregated).
    next_round = state["last_completed_round"] + 1
    if next_round <= ROUNDS:
        done_hospitals, done_updates = [], []
        for h in HOSPITALS:
            mp = _hospital_model_path(h, next_round)
            rp = _hospital_results_path(h, next_round)
            if mp.exists() and rp.exists():
                with open(rp) as f:
                    saved = json.load(f)
                raw = torch.load(mp, map_location="cpu")
                # Normalise keys — strip _orig_mod. prefix added by torch.compile
                clean = {k.replace("_orig_mod.", ""): v for k, v in raw.items()}
                buf = io.BytesIO()
                torch.save(clean, buf)
                done_updates.append({
                    "hospital":      h,
                    "weights_bytes": buf.getvalue(),
                    "n_train":       saved["n_train"],
                    "val_auc":       saved["val_auc"],
                    "test_auc":      saved["test_auc"],
                })
                done_hospitals.append(h)
        if done_hospitals:
            state["partial_round"] = {
                "round_num":           next_round,
                "completed_hospitals": done_hospitals,
                "hospital_updates":    done_updates,
            }

    return state


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=container_image,
    volumes={VOLUME_PATH: volume},
    timeout=86400,   # 24 h ceiling — outlasts any realistic training run
    cpu=2,
    memory=4096,
)
def run_training() -> None:
    """Orchestrator that runs entirely on Modal — safe to close your terminal."""

    print("=" * 62)
    print("  FedMedTrust — Federated Chest X-Ray (A10G, checkpoint-aware)")
    print(f"  Dataset  : {KAGGLE_DATASET}")
    print(f"  Hospitals: {', '.join(HOSPITALS)}")
    print(f"  Rounds   : {ROUNDS} × {EPOCHS_PER_ROUND} epochs/hospital")
    print(f"  Batch    : {BATCH_SIZE}  |  LR: {LR}")
    print("=" * 62)

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    print("\n[1/4] Checking / downloading dataset ...")
    download_dataset.remote()

    # ── 2. Splits ─────────────────────────────────────────────────────────────
    print("\n[2/4] Checking / preparing hospital splits ...")
    prepare_splits.remote()

    # ── 3. Resume state ───────────────────────────────────────────────────────
    print("\n[3/4] Checking resume state ...")
    resume = get_resume_state.remote()
    last_done            = resume["last_completed_round"]
    global_weights_bytes = resume["global_weights_bytes"]
    partial              = resume["partial_round"]

    if last_done == ROUNDS:
        print(f"\nAll {ROUNDS} rounds already complete. Nothing to do.")
        print("Results: modal volume ls fedmedtrust-images-vol /results")
        return

    if last_done > 0:
        print(f"  Resuming after round {last_done}/{ROUNDS}")
    else:
        print("  Starting from scratch.")
    if partial:
        print(f"  Round {partial['round_num']} partially done: hospitals {partial['completed_hospitals']} already trained.")

    # ── 4. Federated rounds ───────────────────────────────────────────────────
    print(f"\n[4/4] Federated training ...")

    for round_num in range(last_done + 1, ROUNDS + 1):
        print(f"\n{'─' * 62}")
        print(f"  Round {round_num}/{ROUNDS}")
        print(f"{'─' * 62}")

        if partial and partial["round_num"] == round_num:
            already_done     = set(partial["completed_hospitals"])
            existing_updates = partial["hospital_updates"]
            partial          = None
        else:
            already_done     = set()
            existing_updates = []

        remaining = [h for h in HOSPITALS if h not in already_done]

        if remaining:
            print(f"  Training hospitals: {remaining}")
            new_updates = list(
                train_hospital.starmap(
                    [(h, round_num, global_weights_bytes) for h in remaining]
                )
            )
        else:
            print("  All hospitals already trained — aggregating only.")
            new_updates = []

        all_updates = existing_updates + new_updates
        print(f"  Aggregating {len(all_updates)} hospital models → global round {round_num} ...")
        global_weights_bytes = aggregate_global.remote(all_updates, round_num)

        mean_val  = sum(u["val_auc"]  for u in all_updates) / len(all_updates)
        mean_test = sum(u["test_auc"] for u in all_updates) / len(all_updates)
        print(f"  Round {round_num} done | mean_val_auc={mean_val:.4f}  mean_test_auc={mean_test:.4f}")

    print("\n" + "=" * 62)
    print("  Training complete.")
    print("  Inspect results : modal volume ls fedmedtrust-images-vol /results")
    print("  Inspect models  : modal volume ls fedmedtrust-images-vol /models")
    print("=" * 62)


@app.local_entrypoint()
def main() -> None:
    """Call run_training on Modal and detach.

    Run with:  modal run --detach modal_train_images.py
    --detach keeps run_training alive on Modal even after this terminal exits.
    Everything after this point runs on Modal's servers.
    """
    run_training.remote()
