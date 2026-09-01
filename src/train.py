"""
Train a YOLO detector on the custom 'computer' dataset.

The dataset was collected from Open Images V7 / Kaggle and augmented in
Roboflow. Export it in YOLOv8 format, which produces the train/valid/test
folders plus a data.yaml describing them.

Usage
-----
    python src/train.py --epochs 150
    python src/train.py --epochs 200 --model yolov9s.pt --patience 100

Notes
-----
Ultralytics enables early stopping by default (patience=100). That is why a
run requested for 200 epochs can terminate at, say, 187: no validation
improvement was observed for 100 consecutive epochs, so training halted and
the best weights were restored. Pass --patience 0 to disable it.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on the computer dataset")
    parser.add_argument("--data", default=str(REPO_ROOT / "data.yaml"),
                        help="Path to the dataset YAML")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="Base checkpoint, e.g. yolov8n.pt, yolov8s.pt, yolov9s.pt")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Training image size in pixels")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size; -1 lets Ultralytics pick automatically")
    parser.add_argument("--patience", type=int, default=100,
                        help="Early-stopping patience; 0 disables it")
    parser.add_argument("--project", default=str(REPO_ROOT / "runs"),
                        help="Directory that will hold the run folders")
    parser.add_argument("--name", default=None,
                        help="Run name; defaults to epochs<N>")
    parser.add_argument("--device", default=None,
                        help="'0' for the first GPU, 'cpu' to force CPU")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed, so runs are comparable")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.name or f"epochs{args.epochs}"

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(
            f"Dataset YAML not found at {data_path}.\n"
            "Download the Roboflow export first, then point --data at its data.yaml."
        )

    print(f"Base model : {args.model}")
    print(f"Dataset    : {data_path}")
    print(f"Epochs     : {args.epochs}  (patience={args.patience})")
    print(f"Run name   : {run_name}\n")

    model = YOLO(args.model)

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=args.project,
        name=run_name,
        seed=args.seed,
        device=args.device,
        exist_ok=True,
        plots=True,      # writes results.png and the confusion matrix
    )

    # Evaluate the best checkpoint on the held-out test split rather than the
    # validation split, so the reported numbers are not the ones early
    # stopping already optimised against.
    metrics = model.val(data=str(data_path), split="test")

    print("\n--- Test split results ---")
    print(f"Precision    : {metrics.box.mp:.4f}")
    print(f"Recall       : {metrics.box.mr:.4f}")
    print(f"mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")

    weights = Path(args.project) / run_name / "weights" / "best.pt"
    print(f"\nBest weights saved to: {weights}")


if __name__ == "__main__":
    main()
