"""Train the carotid segmentation model.

Example:
    python scripts/train.py --data-dir /path/to/carotid-ultrasound-images --output-dir runs/default
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import keras

from carotid_seg.data import SegmentationData, find_pairs, split_pairs
from carotid_seg.evaluation import evaluate
from carotid_seg.model import build_model
from carotid_seg.training import TrainConfig, compile_model, train

logger = logging.getLogger("train")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/default"))
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--val-fraction", type=float, default=TrainConfig.val_fraction)
    parser.add_argument(
        "--no-group-by-scan",
        action="store_true",
        help="Split slices at random instead of keeping each scan on one side.",
    )
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    keras.utils.set_random_seed(args.seed)
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_fraction=args.val_fraction,
        group_by_scan=not args.no_group_by_scan,
        seed=args.seed,
    )

    images, masks = find_pairs(args.data_dir)
    (tr_i, tr_m), (va_i, va_m) = split_pairs(
        images, masks, cfg.val_fraction, cfg.seed, cfg.group_by_scan
    )
    logger.info("%d training and %d validation image/mask pairs", len(tr_i), len(va_i))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "split.json").write_text(
        json.dumps({"train": tr_i, "validation": va_i}, indent=2)
    )

    train_data = SegmentationData(
        tr_i, tr_m, cfg.batch_size, augment=True, shuffle=True, seed=cfg.seed
    )
    val_data = SegmentationData(va_i, va_m, cfg.val_batch_size)
    model = compile_model(build_model(), cfg.learning_rate)
    train(model, train_data, val_data, cfg, args.output_dir)

    result = evaluate(model, val_data)
    (args.output_dir / "validation_metrics.json").write_text(json.dumps(result, indent=2))
    logger.info("Validation Dice (pooled): %.4f", result["pooled"]["dice"])


if __name__ == "__main__":
    main()
