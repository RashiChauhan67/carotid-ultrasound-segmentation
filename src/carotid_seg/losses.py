"""Segmentation losses and metrics for a single foreground class (binary masks)."""

from __future__ import annotations

from keras import ops


def _flat(y):
    return ops.reshape(ops.cast(y, "float32"), (-1,))


def dice_coefficient(y_true, y_pred, smooth: float = 1e-6):
    """Soft Dice coefficient over the whole batch (uses the predicted probabilities)."""
    yt, yp = _flat(y_true), _flat(y_pred)
    return (2.0 * ops.sum(yt * yp) + smooth) / (ops.sum(yt) + ops.sum(yp) + smooth)


def dice_loss(y_true, y_pred, smooth: float = 1e-6):
    return 1.0 - dice_coefficient(y_true, y_pred, smooth)


def combined_loss(y_true, y_pred, smooth: float = 1e-6, alpha: float = 0.5):
    """``alpha * binary cross-entropy + (1 - alpha) * Dice loss``."""
    yt, yp = ops.cast(y_true, "float32"), ops.cast(y_pred, "float32")
    eps = 1e-7
    yp = ops.clip(yp, eps, 1.0 - eps)
    bce = -ops.mean(yt * ops.log(yp) + (1.0 - yt) * ops.log(1.0 - yp))
    return alpha * bce + (1.0 - alpha) * dice_loss(y_true, y_pred, smooth)


def _counts(y_true, y_pred):
    yt, yp = _flat(y_true), ops.round(_flat(y_pred))
    tp = ops.sum(yt * yp)
    fp = ops.sum((1.0 - yt) * yp)
    fn = ops.sum(yt * (1.0 - yp))
    tn = ops.sum((1.0 - yt) * (1.0 - yp))
    return tp, fp, fn, tn


def jaccard_index(y_true, y_pred, smooth: float = 1e-6):
    """Intersection over union of the thresholded (0.5) prediction."""
    tp, fp, fn, _ = _counts(y_true, y_pred)
    return tp / (tp + fp + fn + smooth)


def precision(y_true, y_pred, smooth: float = 1e-6):
    tp, fp, _, _ = _counts(y_true, y_pred)
    return tp / (tp + fp + smooth)


def recall(y_true, y_pred, smooth: float = 1e-6):
    tp, _, fn, _ = _counts(y_true, y_pred)
    return tp / (tp + fn + smooth)


def f1_score(y_true, y_pred, smooth: float = 1e-6):
    p, r = precision(y_true, y_pred, smooth), recall(y_true, y_pred, smooth)
    return 2 * p * r / (p + r + smooth)


def specificity(y_true, y_pred, smooth: float = 1e-6):
    _, fp, _, tn = _counts(y_true, y_pred)
    return tn / (tn + fp + smooth)


METRICS = (dice_coefficient, jaccard_index, precision, recall, f1_score, specificity)
