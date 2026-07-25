"""
preprocessing.py
-----------------
Data acquisition + preprocessing utilities for the Intel Image Classification
(scene classification) dataset: buildings, forest, glacier, mountain, sea, street.

Used by both the training notebook and the retraining pipeline triggered from
the API, so the SAME preprocessing logic is guaranteed to be used at train
time and at retrain time.
"""

import os
import cv2
import numpy as np
from pathlib import Path

IMG_SIZE = 150
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


def download_dataset(dest_dir: str = "data") -> str:
    """
    Downloads the Intel Image Classification dataset from Kaggle using
    kagglehub (requires a free Kaggle account + API token configured as
    described in the README). Returns the path to the extracted dataset.

    NOTE: run this once, locally / on Colab / in your cloud VM -- not needed
    on every retrain, since retraining uses newly uploaded images instead.
    """
    import kagglehub
    path = kagglehub.dataset_download("puneet6060/intel-image-classification")
    print(f"Dataset downloaded to: {path}")
    return path


def load_image(path: str) -> np.ndarray:
    """Read a single image from disk, resize + normalize it to a model-ready array."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    return img


def load_dataset_from_folder(folder: str):
    """
    Expects a folder-of-folders layout:
        folder/buildings/*.jpg
        folder/forest/*.jpg
        ...
    Returns (X, y) as numpy arrays, y as integer class indices.
    """
    X, y = [], []
    folder = Path(folder)
    for class_name in CLASS_NAMES:
        class_dir = folder / class_name
        if not class_dir.exists():
            continue
        for img_path in class_dir.glob("*.*"):
            try:
                X.append(load_image(str(img_path)))
                y.append(CLASS_TO_IDX[class_name])
            except Exception as e:
                print(f"Skipping {img_path}: {e}")
    return np.array(X, dtype="float32"), np.array(y, dtype="int64")


def preprocess_uploaded_images(file_paths: list, labels: list = None):
    """
    Used by the retraining pipeline: preprocess a batch of newly-uploaded
    images (and optional labels) into arrays the model can train on.
    If labels is None, this is being used purely for prediction.
    """
    X = np.array([load_image(p) for p in file_paths], dtype="float32")
    if labels is not None:
        y = np.array([CLASS_TO_IDX[l] if isinstance(l, str) else l for l in labels], dtype="int64")
        return X, y
    return X


def get_image_stats_df(folder: str):
    """
    Builds a per-image dataframe of simple visual features (mean brightness,
    mean R/G/B, image aspect) used for the notebook's exploratory
    visualizations / "what story does the data tell" section.
    """
    import pandas as pd
    rows = []
    folder = Path(folder)
    for class_name in CLASS_NAMES:
        class_dir = folder / class_name
        if not class_dir.exists():
            continue
        for img_path in class_dir.glob("*.*"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rows.append({
                "class": class_name,
                "brightness": img_rgb.mean(),
                "mean_r": img_rgb[:, :, 0].mean(),
                "mean_g": img_rgb[:, :, 1].mean(),
                "mean_b": img_rgb[:, :, 2].mean(),
                "height": img.shape[0],
                "width": img.shape[1],
            })
    return pd.DataFrame(rows)
