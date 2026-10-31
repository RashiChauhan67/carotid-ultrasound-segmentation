import json

import numpy as np

from carotid_seg.data import SegmentationData, find_pairs, split_pairs
from carotid_seg.evaluation import binary_metrics, evaluate
from carotid_seg.model import build_model
from carotid_seg.training import MONITOR, TrainConfig, compile_model, train


def _loaders(root):
    images, masks = find_pairs(root)
    (tr_i, tr_m), (va_i, va_m) = split_pairs(images, masks, val_fraction=0.25)
    return (
        SegmentationData(tr_i, tr_m, batch_size=4, augment=True, shuffle=True, size=(32, 32)),
        SegmentationData(va_i, va_m, batch_size=2, size=(32, 32)),
    )


def test_train_runs_with_a_single_target_and_logs_the_monitored_metric(carotid_root, tmp_path):
    train_data, val_data = _loaders(carotid_root)
    model = compile_model(build_model(input_shape=(32, 32, 3)))
    assert len(model.outputs) == 1  # trained as a single-output mask model

    history = train(model, train_data, val_data, TrainConfig(epochs=1), tmp_path)

    assert MONITOR in history.history
    assert (tmp_path / "best_model.keras").exists() and (tmp_path / "training_log.csv").exists()
    assert json.loads((tmp_path / "config.json").read_text())["epochs"] == 1


def test_evaluate_reports_pooled_and_per_image_metrics(carotid_root):
    _, val_data = _loaders(carotid_root)
    result = evaluate(build_model(input_shape=(32, 32, 3)), val_data)
    assert result["num_images"] == 3
    assert set(result["pooled"]) == {"dice", "iou", "precision", "recall", "specificity"}
    assert 0.0 <= result["per_image_dice_mean"] <= 1.0


def test_binary_metrics_perfect_and_empty_predictions():
    mask = np.zeros((1, 8, 8, 1), dtype="float32")
    mask[0, 2:5, 2:5, 0] = 1.0
    assert binary_metrics(mask, mask)["dice"] > 0.999
    assert binary_metrics(mask, np.zeros_like(mask))["dice"] < 1e-6
