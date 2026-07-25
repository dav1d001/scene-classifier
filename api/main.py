"""
api/main.py
-----------
FastAPI backend exposing:
  GET  /health              -> liveness probe
  GET  /uptime               -> uptime + request-count stats for the UI
  POST /predict               -> single-image prediction
  POST /upload-retrain-data   -> bulk image upload for retraining (labeled)
  POST /retrain                -> manually trigger a retraining job
  GET  /retrain/status         -> status/history of retrain jobs
  GET  /metrics/live            -> latest production model metrics
"""

import os
import time
import shutil
import threading
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import CLASS_NAMES
from src.prediction import predict_single_image
from src.retrain import run_retrain_job, should_auto_retrain, count_pending_uploads, UPLOAD_DIR
from src.model import load_model, MODEL_PATH
from api.database import log_prediction, log_retrain_job, get_recent_predictions, get_retrain_history

app = FastAPI(title="Scene Classifier API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

START_TIME = time.time()
_model_lock = threading.Lock()
_model = None
_retrain_running = False


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            if not os.path.exists(MODEL_PATH):
                raise HTTPException(status_code=503, detail="Model not trained/deployed yet.")
            _model = load_model(MODEL_PATH)
        return _model


def reload_model():
    """Called after a successful retrain so the API serves the new weights
    without needing a process restart."""
    global _model
    with _model_lock:
        _model = load_model(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/uptime")
def uptime():
    seconds = time.time() - START_TIME
    return {
        "uptime_seconds": round(seconds, 1),
        "uptime_human": _human_readable(seconds),
        "started_at": datetime.fromtimestamp(START_TIME).isoformat(),
        "model_loaded": _model is not None,
        "pending_retrain_images": count_pending_uploads(),
    }


def _human_readable(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return f"{d}d {h}h {m}m {s}s"


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    os.makedirs("data/tmp", exist_ok=True)
    tmp_path = f"data/tmp/{int(time.time()*1000)}_{file.filename}"
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    start = time.time()
    try:
        model = get_model()
        result = predict_single_image(model, tmp_path)
    finally:
        os.remove(tmp_path)
    latency_ms = round((time.time() - start) * 1000, 2)

    log_prediction(result["predicted_class"], result["confidence"], latency_ms)
    result["latency_ms"] = latency_ms
    return result


@app.post("/upload-retrain-data")
async def upload_retrain_data(label: str = Form(...), files: list[UploadFile] = File(...)):
    if label not in CLASS_NAMES:
        raise HTTPException(status_code=400, detail=f"label must be one of {CLASS_NAMES}")

    dest_dir = os.path.join(UPLOAD_DIR, label)
    os.makedirs(dest_dir, exist_ok=True)
    saved = []
    for f in files:
        dest_path = os.path.join(dest_dir, f"{int(time.time()*1000)}_{f.filename}")
        with open(dest_path, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(dest_path)

    return {
        "saved": len(saved),
        "label": label,
        "total_pending": count_pending_uploads(),
        "auto_retrain_ready": should_auto_retrain(),
    }


def _background_retrain():
    global _retrain_running
    _retrain_running = True
    try:
        report = run_retrain_job()
        log_retrain_job(report)
        if report.get("deployed"):
            reload_model()
    finally:
        _retrain_running = False


@app.post("/retrain")
def trigger_retrain(background_tasks: BackgroundTasks):
    if _retrain_running:
        return {"status": "already_running"}
    if count_pending_uploads() < 10:
        raise HTTPException(status_code=400, detail="Need at least 10 new labeled images to retrain.")
    background_tasks.add_task(_background_retrain)
    return {"status": "started"}


@app.get("/retrain/status")
def retrain_status():
    return {
        "running": _retrain_running,
        "pending_images": count_pending_uploads(),
        "history": get_retrain_history(limit=20),
    }


@app.get("/metrics/live")
def live_metrics():
    return {
        "recent_predictions": get_recent_predictions(limit=50),
    }
