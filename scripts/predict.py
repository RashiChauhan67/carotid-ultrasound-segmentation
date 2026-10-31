"""Segment the carotid artery in ultrasound images and save masks and overlays.

Example:
    python scripts/predict.py --model runs/default/best_model.keras --images path/to/images
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import keras
import numpy as np

import carotid_seg.model  # noqa: F401  (registers the custom layer for loading)
from carotid_seg.data import IMAGE_SIZE


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--images", type=Path, required=True, help="An image or a folder of PNG/JPG images."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/predict"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    model = keras.models.load_model(args.model, compile=False)
    paths = (
        [args.images]
        if args.images.is_file()
        else sorted(
            p for p in args.images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        bgr = cv2.imread(str(path))
        small = (
            cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), IMAGE_SIZE).astype("float32") / 255.0
        )
        out = model.predict(small[None], verbose=0)
        out = out["segmentation"] if isinstance(out, dict) else out
        prob = out[0, ..., 0]
        mask = (prob >= args.threshold).astype(np.uint8) * 255
        overlay = cv2.resize(bgr, IMAGE_SIZE)
        overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.imwrite(str(args.output_dir / f"{path.stem}_mask.png"), mask)
        cv2.imwrite(str(args.output_dir / f"{path.stem}_overlay.png"), overlay)
    print(f"Saved {len(paths)} masks and overlays to {args.output_dir}")


if __name__ == "__main__":
    main()
