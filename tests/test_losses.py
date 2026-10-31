import numpy as np
import pytest

from carotid_seg.evaluation import binary_metrics
from carotid_seg.losses import (
    combined_loss,
    dice_coefficient,
    dice_loss,
    f1_score,
    jaccard_index,
    precision,
    recall,
    specificity,
)


def _mask(rows, cols, size=8):
    m = np.zeros((1, size, size, 1), dtype="float32")
    m[0, rows[0] : rows[1], cols[0] : cols[1], 0] = 1.0
    return m


def test_dice_is_one_for_a_perfect_prediction_and_near_zero_for_disjoint_masks():
    a = _mask((1, 4), (1, 4))
    assert float(dice_coefficient(a, a)) == pytest.approx(1.0, abs=1e-5)
    assert float(dice_coefficient(a, _mask((5, 8), (5, 8)))) < 1e-4
    assert float(dice_loss(a, a)) == pytest.approx(0.0, abs=1e-5)


def test_combined_loss_is_the_weighted_sum_of_bce_and_dice():
    y = _mask((1, 4), (1, 4))
    p = np.clip(y * 0.8 + 0.1, 0, 1)
    bce = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    expected = 0.5 * bce + 0.5 * float(dice_loss(y, p))
    assert float(combined_loss(y, p)) == pytest.approx(expected, rel=1e-4)


def test_thresholded_metrics_on_a_known_case():
    y_true = _mask((0, 4), (0, 4))  # 16 px
    y_pred = _mask((0, 4), (0, 2))  # 8 px, all inside the label
    assert float(jaccard_index(y_true, y_pred)) == pytest.approx(8 / 16, abs=1e-5)
    assert float(precision(y_true, y_pred)) == pytest.approx(1.0, abs=1e-5)
    assert float(recall(y_true, y_pred)) == pytest.approx(0.5, abs=1e-5)
    assert float(f1_score(y_true, y_pred)) == pytest.approx(2 / 3, abs=1e-4)
    assert float(specificity(y_true, y_pred)) == pytest.approx(1.0, abs=1e-5)


def test_numpy_metrics_agree_with_the_keras_metrics():
    y_true = _mask((0, 4), (0, 4))
    y_pred = _mask((0, 4), (0, 2))
    np_metrics = binary_metrics(y_true, y_pred)
    assert np_metrics["iou"] == pytest.approx(float(jaccard_index(y_true, y_pred)), abs=1e-4)
    assert np_metrics["recall"] == pytest.approx(float(recall(y_true, y_pred)), abs=1e-4)
