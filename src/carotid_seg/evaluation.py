"""Evaluate predicted masks against expert masks."""

from __future__ import annotations

import keras
import numpy as np

from .data import SegmentationData


def binary_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Dice, IoU, precision, recall and specificity pooled over all pixels."""
    t = np.asarray(y_true).reshape(-1) > 0.5
    p = np.asarray(y_prob).reshape(-1) >= threshold
    tp, fp = int(np.sum(t & p)), int(np.sum(~t & p))
    fn, tn = int(np.sum(t & ~p)), int(np.sum(~t & ~p))
    eps = 1e-9
    return {
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "specificity": tn / (tn + fp + eps),
    }


def evaluate(model: keras.Model, data: SegmentationData, threshold: float = 0.5) -> dict:
    """Predict every batch of ``data`` and report pooled metrics plus per-image Dice statistics."""
    truths, probs = [], []
    for i in range(len(data)):
        images, targets = data[i]
        outputs = model.predict(images, verbose=0)
        if isinstance(outputs, dict):
            segmentation = outputs["segmentation"]
        elif isinstance(outputs, (list, tuple)):
            segmentation = outputs[0]
        else:
            segmentation = outputs
        truths.append(targets)
        probs.append(segmentation)
    y_true, y_prob = np.concatenate(truths), np.concatenate(probs)

    per_image = np.array(
        [binary_metrics(t, p, threshold)["dice"] for t, p in zip(y_true, y_prob, strict=True)]
    )
    return {
        "num_images": int(len(y_true)),
        "pooled": binary_metrics(y_true, y_prob, threshold),
        "per_image_dice_mean": float(per_image.mean()),
        "per_image_dice_std": float(per_image.std()),
    }
