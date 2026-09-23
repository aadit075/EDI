"""CLI utility to predict diseases from chest X-ray images using trained FedMedTrust models."""
import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output if possible
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from fedmedtrust_images.predict_image import predict_xray_from_bytes, DISEASE_LABELS

def format_bar(prob: float, width: int = 20) -> str:
    filled = int(round(prob * width))
    return "#" * filled + "-" * (width - filled)

def predict_single_image(image_path: Path, hospital: str = "A"):
    if not image_path.exists():
        print(f"Error: Image {image_path} does not exist.")
        return None

    with open(image_path, "rb") as f:
        img_bytes = f.read()

    preds = predict_xray_from_bytes(hospital, img_bytes)
    sorted_preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)

    print("=" * 66)
    print(f" Image    : {image_path.name}")
    print(f" Hospital : {hospital} (ResNet18 federated model)")
    print("=" * 66)

    cardiomegaly = preds.get("Cardiomegaly", 0.0)
    if cardiomegaly >= 0.3:
        status = "NOTABLE (High Risk)" if cardiomegaly >= 0.5 else "BORDERLINE"
        print(f"\n  [!] CARDIOMEGALY DETECTED: {cardiomegaly*100:.1f}% [{status}]")
        print("      Enlarged heart detected - corroborates cardiac risk assessment.\n")

    print(f"  {'Disease Finding':<22} {'Prob':>6}   {'Status':<8}  {'Distribution'}")
    print("  " + "-" * 62)
    for disease, prob in sorted_preds:
        pct = prob * 100
        bar = format_bar(prob)
        flag = "[HIGH]" if pct >= 50 else ("[MOD] " if pct >= 30 else "[LOW] ")
        print(f"  {disease.replace('_', ' '):<22} {pct:5.1f}%   {flag:<8}  [{bar}]")
    print("=" * 66)
    return preds

def main():
    parser = argparse.ArgumentParser(description="Predict diseases from chest X-ray images.")
    parser.add_argument("image", nargs="?", default=None, help="Path to X-ray image file")
    parser.add_argument("--hospital", "-H", default="A", choices=["A", "B", "C", "D", "E"], help="Hospital branch (A-E)")
    parser.add_argument("--all-samples", action="store_true", help="Run inference on all sample images in sample_xrays/")
    args = parser.parse_args()

    if args.all_samples:
        sample_dir = Path("sample_xrays")
        images = sorted([p for p in sample_dir.glob("*.png")])
        print(f"\nRunning disease predictions for {len(images)} sample X-ray images using Hospital {args.hospital} model:\n")
        for p in images:
            predict_single_image(p, hospital=args.hospital)
            print()
    elif args.image:
        predict_single_image(Path(args.image), hospital=args.hospital)
    else:
        sample_dir = Path("sample_xrays")
        first_img = next(sample_dir.glob("*.png"), None)
        if first_img:
            print(f"\nNo image specified. Predicting first sample image: {first_img}\n")
            predict_single_image(first_img, hospital=args.hospital)
        else:
            parser.print_help()

if __name__ == "__main__":
    main()
