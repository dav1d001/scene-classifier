"""
retrain.py
----------
Retraining pipeline: takes newly uploaded, labeled images, preprocesses them
with the SAME logic used at initial training time, fine-tunes the existing
saved model on them, evaluates the new model, and only replaces the
production model file if it does not regress accuracy.

Also defines an automatic trigger condition: if enough new images pile up
(default 50) OR the user presses "Retrain now" in the UI, a retrain job runs.
"""

import os
import shutil
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

from src.preprocessing import preprocess_uploaded_images, CLASS_NAMES
from src.model import load_model, train_model, evaluate_model, save_model, MODEL_PATH

UPLOAD_DIR = "data/uploads"          # where bulk-uploaded retrain images land
RETRAIN_THRESHOLD = 50               # auto-trigger once this many new images accumulate


def count_pending_uploads() -> int:
    if not os.path.isdir(UPLOAD_DIR):
        return 0
    total = 0
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(UPLOAD_DIR, cls)
        if os.path.isdir(cls_dir):
            total += len([f for f in os.listdir(cls_dir) if not f.startswith(".")])
    return total


def should_auto_retrain() -> bool:
    return count_pending_uploads() >= RETRAIN_THRESHOLD


def run_retrain_job(min_accuracy_to_deploy: float = 0.0) -> dict:
    """
    Executes one retraining cycle:
      1. Preprocess every image currently sitting in data/uploads/<class>/
      2. Fine-tune the current production model on this new data (+ a small
         held-out split for evaluation)
      3. Evaluate the retrained model
      4. Only overwrite the production model if the new accuracy is >= the
         previous production accuracy (or >= min_accuracy_to_deploy), so a
         bad batch of uploads can't silently degrade the live model
      5. Archive the processed images so they aren't retrained on again
    Returns a job report dict (also what the API/UI display).
    """
    file_paths, labels = [], []
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(UPLOAD_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.startswith("."):
                continue
            file_paths.append(os.path.join(cls_dir, fname))
            labels.append(cls)

    if len(file_paths) < 10:
        return {"status": "skipped", "reason": "Fewer than 10 new labeled images available."}

    X, y = preprocess_uploaded_images(file_paths, labels)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    current_model = load_model(MODEL_PATH)
    pre_retrain_metrics = evaluate_model(current_model, X_val, y_val)

    retrained_model, history = train_model(
        X_train, y_train, X_val, y_val,
        epochs=8, batch_size=16, model=current_model, fine_tune=True,
    )
    post_retrain_metrics = evaluate_model(retrained_model, X_val, y_val)

    deployed = False
    threshold = max(min_accuracy_to_deploy, pre_retrain_metrics["accuracy"])
    if post_retrain_metrics["accuracy"] >= threshold:
        backup_path = MODEL_PATH.replace(".keras", f"_backup_{datetime.now():%Y%m%d%H%M%S}.keras")
        if os.path.exists(MODEL_PATH):
            shutil.copy(MODEL_PATH, backup_path)
        save_model(retrained_model, MODEL_PATH)
        deployed = True
        _archive_processed_uploads(file_paths)

    return {
        "status": "completed",
        "deployed": deployed,
        "n_images_used": len(file_paths),
        "accuracy_before": pre_retrain_metrics["accuracy"],
        "accuracy_after": post_retrain_metrics["accuracy"],
        "f1_after": post_retrain_metrics["f1_macro"],
        "timestamp": datetime.now().isoformat(),
    }


def _archive_processed_uploads(file_paths):
    archive_dir = "data/uploads_archive"
    for p in file_paths:
        rel = os.path.relpath(p, UPLOAD_DIR)
        dest = os.path.join(archive_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(p, dest)
