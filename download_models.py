"""Download trained chest X-ray models to outputs/image_models/.

Models are hosted on Hugging Face Hub at:
    https://huggingface.co/ththjyj755/fedmedtrust-image-models

No Modal account required.  Only needs:
    pip install huggingface_hub

Run once after cloning:
    python download_models.py
"""
import sys
from pathlib import Path

REPO_ID = "ththjyj755/fedmedtrust-image-models"
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
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub not found. Install it with:")
        print("    pip install huggingface_hub")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading models from Hugging Face ({REPO_ID}) ...\n")

    for name in MODELS:
        dest = OUT_DIR / name
        if dest.exists():
            print(f"  already present — {name}")
            continue
        print(f"  downloading {name} ...", end=" ", flush=True)
        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=name,
                local_dir=str(OUT_DIR),
            )
            size_mb = Path(path).stat().st_size / 1_048_576
            print(f"done ({size_mb:.1f} MB)")
        except Exception as exc:
            print(f"FAILED — {exc}")
            sys.exit(1)

    print(f"\nAll models saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
