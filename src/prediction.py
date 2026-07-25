"""
prediction.py
-------------
Single-datapoint inference, used by both the API's /predict endpoint and the
notebook's "model testing" section.
"""

import numpy as np
from src.preprocessing import load_image, CLASS_NAMES


def predict_single_image(model, image_path: str) -> dict:
    """Runs inference on one image and returns the predicted class + confidence
    scores for every class (useful for showing model certainty in the UI)."""
    img = load_image(image_path)
    img_batch = np.expand_dims(img, axis=0)
    probs = model.predict(img_batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_scores": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
    }
