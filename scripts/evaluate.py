"""Evaluate a saved model on the validation split of a dataset.

The split is regenerated from the data folder with the same seed.

Example:
    python scripts/evaluate.py --data-dir /path/to/carotid-ultrasound-images \\
        --model runs/default/best_model.keras
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras

import carotid_seg.model  # noqa: F401  (registers the custom layer for loading)
from carotid_seg.data import SPLIT_SEED, SegmentationData, find_pairs, split_pairs
from carotid_seg.evaluation import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--no-group-by-scan", action="store_true")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    images, masks = find_pairs(args.data_dir)
    _, (va_i, va_m) = split_pairs(
        images, masks, args.val_fraction, args.seed, not args.no_group_by_scan
    )
    model = keras.models.load_model(args.model, compile=False)
    print(json.dumps(evaluate(model, SegmentationData(va_i, va_m, 4)), indent=2))


if __name__ == "__main__":
    main()
