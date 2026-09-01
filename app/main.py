"""
FastAPI service for the computer detector.

Upload an image, get it back with bounding boxes drawn on it, plus a JSON
summary of what was found.

Run
---
    uvicorn app.main:app --reload
    # then open http://127.0.0.1:8000

Environment
-----------
    WEIGHTS_PATH   override the checkpoint (default: runs/epochs150/weights/best.pt)
    CONF_THRESHOLD override the confidence threshold (default: 0.35)
"""

import base64
import io
import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]

WEIGHTS = Path(os.getenv("WEIGHTS_PATH",
                         REPO_ROOT / "runs" / "epochs150" / "weights" / "best.pt"))
CONF = float(os.getenv("CONF_THRESHOLD", "0.35"))
MAX_UPLOAD_BYTES = 10 * 1024 * 1024   # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

app = FastAPI(title="Computer Detection API", version="1.0.0")
templates = Jinja2Templates(directory=str(REPO_ROOT / "app" / "templates"))

# Loaded once at import time. Reloading per request would add seconds of
# latency to every upload for no benefit.
model: YOLO | None = None


@app.on_event("startup")
def load_model() -> None:
    global model
    if not WEIGHTS.exists():
        print(f"WARNING: weights not found at {WEIGHTS}. "
              f"Train a model first, or set WEIGHTS_PATH.")
        return
    model = YOLO(str(WEIGHTS))
    print(f"Model loaded from {WEIGHTS}")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "model_ready": model is not None},
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "weights": str(WEIGHTS)}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, "Model not loaded. Train a model first.")

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image too large (10 MB limit).")

    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode that file as an image.")

    frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    results = model.predict(source=frame, conf=CONF, verbose=False)
    result = results[0]

    detections = [
        {
            "label": model.names[int(box.cls)],
            "confidence": round(float(box.conf), 3),
            "box": [round(v, 1) for v in box.xyxy[0].tolist()],
        }
        for box in result.boxes
    ]

    # result.plot() returns the annotated frame as a BGR numpy array.
    annotated = result.plot()
    ok, buffer = cv2.imencode(".jpg", annotated)
    if not ok:
        raise HTTPException(500, "Failed to encode the annotated image.")

    return JSONResponse({
        "count": len(detections),
        "detections": detections,
        "image": "data:image/jpeg;base64,"
                 + base64.b64encode(buffer.tobytes()).decode("utf-8"),
    })
