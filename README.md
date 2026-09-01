# Computer Detection with YOLO

A real-time object detector trained to find computers in photographs, with a
comparative study of how training length affects detection quality, and a
FastAPI web app for running the model on new images.

> **Group project** — BEJ30303 Computer Architecture and Organization,
> Universiti Tun Hussein Onn Malaysia, Semester 1 2024/2025.
> Team of 4. My contribution: dataset curation and augmentation setup, model training runs, and the deployment application.
---

## What this does

Given an image, the model locates every computer in it and draws a bounding
box around each one with a confidence score.

The more interesting part is the training study. A model trained for 100
epochs underfits. A model trained for 200 starts memorising the training set
instead of generalising. Somewhere between the two is the configuration you
actually want, and `compare_epochs.py` exists to find it with numbers rather
than intuition.

## Dataset

| | |
|---|---|
| Source | Open Images V7 and Kaggle |
| Annotation and augmentation | Roboflow |
| Total images | 1,152 |
| Train / Validation / Test | 957 / 94 / 83 |
| Image size | 640 × 640 |
| Classes | 1 (`computer`) |

**Preprocessing:** auto-orient from EXIF metadata, adaptive contrast
equalisation, resize to 640 × 640.

**Augmentation:** three variants generated per training image using crop zoom
(0–10%) and bounding box position shift (±10%), taking 957 source images to
2,871 training examples.

The dataset is not committed to this repository. Export it from Roboflow in
YOLOv8 format into a `dataset/` folder at the repository root.

## A note on weights

No trained checkpoint ships with this repository. Weights are large, and the
ones from the original study are no longer on hand.

What is here is the full pipeline: training, the comparative epoch study,
batch inference, and the web app. Point it at your own dataset and it will
produce a model. The findings below came from the original run and are
recorded so you know roughly what to expect, not as numbers to trust blindly.

Training is slow. On a Colab T4, a single 150-epoch run on this dataset takes
somewhere around 1.5 hours, and the three-budget comparison takes most of an
evening. Plan accordingly, and use `--skip-training` on `compare_epochs.py`
once the runs exist so you are not repeating work.

## Setup

```bash
git clone <this-repo>
cd yolo-computer-detection

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place the Roboflow export at `dataset/`, so the layout looks like:

```
dataset/
├── train/{images,labels}
├── valid/{images,labels}
├── test/{images,labels}
└── data.yaml
```

## Usage

**Train a single model**

```bash
python src/train.py --epochs 150
```

**Run the epoch comparison study**

```bash
python src/compare_epochs.py --budgets 100 150 200
```

Writes `reports/epoch_comparison.csv` and `reports/epoch_comparison.png`.
If the runs already exist on disk, add `--skip-training` to just read them.

**Detect on new images**

```bash
python src/predict.py --source path/to/images --conf 0.35
```

**Run the web app**

```bash
uvicorn app.main:app --reload
# http://127.0.0.1:8000
```

## Findings

From the original study, across three budgets:

150 epochs gave the best balance of precision against validation loss.
Pushing to 200 lowered training loss slightly but precision stopped
improving, which is the signature of a model beginning to memorise rather
than generalise. Early stopping halted that run at 187 epochs after 100
epochs without validation improvement. At 100 epochs the loss was still
falling, so the model had not finished learning.

Absolute numbers depend on the GPU, the seed, and the exact dataset export,
so treat the shape of the result as the finding, not the decimals.

## Known limitations

**Confusion between computers, monitors and desktops.** Every training image
contains a computer, so the model never learned what a near-miss looks like.
It flags standalone monitors as computers with fair confidence. The fix is
hard negatives: images of monitors, televisions and empty desks labelled as
background.

**Single class.** Separating `laptop`, `desktop` and `monitor` into distinct
classes would make the model more useful and force it to learn the boundaries
it currently blurs.

**Narrow angle coverage.** Most source images are product-style photographs
shot straight on. Detection degrades on oblique angles and poor lighting.

## Structure

```
├── src/
│   ├── train.py            single training run
│   ├── compare_epochs.py   multi-budget comparison study
│   └── predict.py          batch inference
├── app/
│   ├── main.py             FastAPI service
│   └── templates/
│       └── index.html      upload interface
├── data.yaml               dataset configuration
└── requirements.txt
```

## License

MIT
