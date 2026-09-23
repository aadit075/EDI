"""Download trained image models from Modal volume to outputs/image_models/.

Run once after cloning:
    python download_models.py
"""
import subprocess
import sys
from pathlib import Path

VOLUME  = "fedmedtrust-images-vol"
MODELS  = [
    "global_model_latest.pt",
    "hospital_A_round2.pt",
    "hospital_B_round2.pt",
    "hospital_C_round2.pt",
    "hospital_D_round2.pt",
    "hospital_E_round2.pt",
]
OUT_DIR = Path(__file__).parent / "outputs" / "image_models"

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        dest = OUT_DIR / name
        if dest.exists():
            print(f"  already present — {name}")
            continue
        print(f"  downloading {name} ...", end=" ", flush=True)
        result = subprocess.run(
            ["modal", "volume", "get", VOLUME, f"/models/{name}", str(dest)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            size_mb = dest.stat().st_size / 1_048_576
            print(f"done ({size_mb:.1f} MB)")
        else:
            print(f"FAILED\n{result.stderr.strip()}")
            sys.exit(1)
    print(f"\nAll models saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
