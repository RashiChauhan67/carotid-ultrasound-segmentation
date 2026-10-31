"""Compile and train the segmentation model.

Training uses a single-output view of the network (the mask). The ``edge_pred`` output has no
label of its own: it is an internal edge map fed back into the decoder (see ``model.py``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import keras

from .data import SPLIT_SEED, SegmentationData
from .losses import METRICS, combined_loss
from .model import segmentation_model

MONITOR = "val_dice_coefficient"


@dataclass
class TrainConfig:
    """Training settings. Defaults match the notebook, except where noted in the README."""

    epochs: int = 100
    batch_size: int = 8
    val_batch_size: int = 4
    learning_rate: float = 1e-4
    patience: int = 20
    lr_patience: int = 10
    val_fraction: float = 0.2
    group_by_scan: bool = True
    seed: int = SPLIT_SEED


def compile_model(model: keras.Model, learning_rate: float = 1e-4) -> keras.Model:
    """Compile for training and return the model to fit.

    A two-output network is reduced to its single-output mask view first. Loss is BCE + Dice and
    the metrics are overlap measures, all on the mask.
    """
    if len(model.outputs) > 1:
        model = segmentation_model(model)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=combined_loss,
        metrics=list(METRICS),
    )
    return model


def make_callbacks(cfg: TrainConfig, output_dir: Path | None) -> list:
    """Early stopping, learning-rate reduction and (optionally) checkpoint + CSV log."""
    callbacks: list = [
        keras.callbacks.EarlyStopping(
            monitor=MONITOR, mode="max", patience=cfg.patience, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=MONITOR, mode="max", factor=0.2, patience=cfg.lr_patience, min_lr=1e-6
        ),
    ]
    if output_dir is not None:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                str(output_dir / "best_model.keras"),
                monitor=MONITOR,
                mode="max",
                save_best_only=True,
            )
        )
        callbacks.append(keras.callbacks.CSVLogger(str(output_dir / "training_log.csv")))
    return callbacks


def train(
    model: keras.Model,
    train_data: SegmentationData,
    val_data: SegmentationData,
    cfg: TrainConfig | None = None,
    output_dir: Path | None = None,
) -> keras.callbacks.History:
    """Fit the model; weights of the best validation Dice are restored at the end."""
    cfg = cfg or TrainConfig()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    return model.fit(
        train_data,
        validation_data=val_data,
        epochs=cfg.epochs,
        callbacks=make_callbacks(cfg, output_dir),
        verbose=2,
    )
