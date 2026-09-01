"""
Run the trained detector over a folder of images and save annotated copies.

Usage
-----
    python src/predict.py --source data/test_images --conf 0.35
    python src/predict.py --source photo.jpg --weights runs/epochs150/weights/best.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "runs" / "epochs150" / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect computers in images")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--source", required=True,
                        help="Image file, folder of images, or video")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Confidence threshold; raise it to cut false positives")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold; lower it when boxes overlap too much")
    parser.add_argument("--out", default=str(REPO_ROOT / "outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(
            f"Weights not found at {weights}.\n"
            "Train a model first with: python src/train.py --epochs 150"
        )

    model = YOLO(str(weights))
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save=True,
        project=args.out,
        name="predict",
        exist_ok=True,
    )

    total = 0
    for r in results:
        n = len(r.boxes)
        total += n
        print(f"{Path(r.path).name}: {n} detection(s)")

    print(f"\n{total} detection(s) across {len(results)} image(s).")
    print(f"Annotated images saved under {Path(args.out) / 'predict'}")


if __name__ == "__main__":
    main()
