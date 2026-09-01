"""
Compare model quality across several epoch budgets.

This is the heart of the study. Training longer is not automatically better:
too few epochs and the model underfits, too many and it starts memorising the
training set instead of learning what a computer looks like. This script runs
the same configuration at different epoch counts and puts the numbers side by
side so the trade-off is visible rather than asserted.

Usage
-----
    python src/compare_epochs.py --budgets 100 150 200
    python src/compare_epochs.py --budgets 100 150 200 --skip-training

--skip-training reads results from runs that already exist on disk, which is
useful when training happened earlier in a Colab session.

Outputs
-------
    reports/epoch_comparison.csv
    reports/epoch_comparison.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"

# Column names written by Ultralytics into results.csv.
COLS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
    "val_box_loss": "val/box_loss",
    "val_cls_loss": "val/cls_loss",
    "val_dfl_loss": "val/dfl_loss",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare epoch budgets")
    parser.add_argument("--budgets", type=int, nargs="+", default=[100, 150, 200],
                        help="Epoch counts to compare")
    parser.add_argument("--data", default=str(REPO_ROOT / "data.yaml"))
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--project", default=str(REPO_ROOT / "runs"))
    parser.add_argument("--skip-training", action="store_true",
                        help="Only read existing runs, do not train")
    return parser.parse_args()


def train_budget(args: argparse.Namespace, epochs: int) -> None:
    from ultralytics import YOLO  # imported lazily so --skip-training is light

    print(f"\n{'=' * 60}\nTraining for {epochs} epochs\n{'=' * 60}")
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=epochs,
        imgsz=640,
        project=args.project,
        name=f"epochs{epochs}",
        seed=42,
        exist_ok=True,
        plots=True,
    )


def read_run(project: Path, epochs: int) -> dict | None:
    """Pull the best-epoch row out of one run's results.csv."""
    results_csv = project / f"epochs{epochs}" / "results.csv"
    if not results_csv.exists():
        print(f"  skipped: {results_csv} not found")
        return None

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    # "Best" is the epoch with the highest mAP@0.5:0.95, which is the metric
    # Ultralytics itself uses to decide which checkpoint to keep.
    best_idx = df[COLS["map50_95"]].idxmax()
    best = df.loc[best_idx]

    row = {
        "requested_epochs": epochs,
        "epochs_completed": int(df.shape[0]),
        "best_epoch": int(best.get("epoch", best_idx + 1)),
    }
    for label, col in COLS.items():
        row[label] = round(float(best[col]), 4) if col in df.columns else None

    # Total validation loss makes the underfit/overfit story easier to read
    # than three separate loss curves.
    losses = [row[k] for k in ("val_box_loss", "val_cls_loss", "val_dfl_loss")
              if row.get(k) is not None]
    row["val_loss_total"] = round(sum(losses), 4) if losses else None
    return row


def plot(rows: list[dict], out_path: Path) -> None:
    budgets = [r["requested_epochs"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(budgets, [r["precision"] for r in rows], marker="o", label="Precision")
    ax1.plot(budgets, [r["recall"] for r in rows], marker="s", label="Recall")
    ax1.plot(budgets, [r["map50"] for r in rows], marker="^", label="mAP@0.5")
    ax1.set_xlabel("Requested epochs")
    ax1.set_ylabel("Score")
    ax1.set_title("Detection quality vs training length")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(budgets, [r["val_loss_total"] for r in rows],
             marker="o", color="crimson")
    ax2.set_xlabel("Requested epochs")
    ax2.set_ylabel("Total validation loss")
    ax2.set_title("Validation loss vs training length")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Chart written to {out_path}")


def main() -> None:
    args = parse_args()
    project = Path(args.project)

    if not args.skip_training:
        for epochs in args.budgets:
            train_budget(args, epochs)

    rows = []
    for epochs in args.budgets:
        row = read_run(project, epochs)
        if row:
            rows.append(row)

    if not rows:
        raise SystemExit("No runs found. Train first, or check --project.")

    REPORTS.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS / "epoch_comparison.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nTable written to {csv_path}")

    plot(rows, REPORTS / "epoch_comparison.png")

    print("\n--- Summary ---")
    header = f"{'Epochs':>8} {'Done':>6} {'Prec':>8} {'Recall':>8} {'mAP50':>8} {'ValLoss':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['requested_epochs']:>8} {r['epochs_completed']:>6} "
              f"{r['precision']:>8.4f} {r['recall']:>8.4f} "
              f"{r['map50']:>8.4f} {r['val_loss_total']:>9.4f}")

    best = max(rows, key=lambda r: r["map50"])
    print(f"\nHighest mAP@0.5 at a budget of {best['requested_epochs']} epochs.")
    print("Read this alongside validation loss: if loss rises while precision "
          "stays flat, the extra epochs bought memorisation, not skill.")


if __name__ == "__main__":
    main()
