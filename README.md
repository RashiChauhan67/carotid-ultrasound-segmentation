# Carotid Ultrasound Segmentation

**A lightweight attention network that segments the common carotid artery in ultrasound images.**

[![CI](https://github.com/RashiChauhan67/carotid-ultrasound-segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/RashiChauhan67/carotid-ultrasound-segmentation/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![TensorFlow 2.16+](https://img.shields.io/badge/TensorFlow-2.16%2B-orange)

[Overview](#overview) · [Architecture](#architecture) · [Quick start](#quick-start) · [Dataset](#dataset) · [Pipeline](#pipeline) · [Notes](#notes)

## Overview

The network takes a 256 x 256 ultrasound image and predicts a binary mask of the artery cross-section. It combines a depthwise-separable encoder, an ASPP-like multi-scale context module, attention gates on the skip connections, squeeze-and-excitation, coordinate attention, and a boundary-refinement branch whose edge map is fed back into the decoder.

This repository has a cleaned, tested Keras 3 implementation of the model and a leakage-free training pipeline, plus the original experiment notebook.

| | |
|---|---|
| Task | binary segmentation (artery mask) |
| Input | 256 x 256 x 3 |
| Model | 171,653 parameters (169,887 trainable) |
| Loss | binary cross-entropy + Dice |
| Metrics | Dice, IoU, precision, recall, F1, specificity |
| Dataset | 1,100 ultrasound image / expert-mask pairs |

## Architecture

![Carotid segmentation architecture](docs/figures/carotid_architecture.png)

Decoder block:

![Decoder block](docs/figures/carotid_decoder_block.png)

## Quick start

```bash
git clone https://github.com/RashiChauhan67/carotid-ultrasound-segmentation.git
cd carotid-ultrasound-segmentation
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Train, evaluate and predict:

```bash
# Train (up to 100 epochs, batch size 8, Adam 1e-4, early stopping on validation Dice)
python scripts/train.py --data-dir /path/to/carotid-ultrasound-images --output-dir runs/default

# Evaluate a saved model on the regenerated validation split
python scripts/evaluate.py --data-dir /path/to/carotid-ultrasound-images \
    --model runs/default/best_model.keras

# Segment new images; saves masks and overlays
python scripts/predict.py --model runs/default/best_model.keras --images path/to/images
```

Training writes `best_model.keras`, `training_log.csv`, `config.json`, `split.json` (the exact train/validation files) and `validation_metrics.json`.

Use the model directly:

```python
from carotid_seg.model import build_model
from carotid_seg.training import compile_model

model = compile_model(build_model())  # 171,653 parameters; trained as a mask-only model
```

## Dataset

Common carotid artery ultrasound images with expert masks: 1,100 image / mask pairs (`US images/` and `Expert mask images/`, matched by file name). Data is not included; see [`data/README.md`](data/README.md).

![Ultrasound images and expert masks](results/figures/dataset_image_mask_pairs.png)

Mask overlays on ultrasound images:

![Mask overlays](results/figures/dataset_mask_overlays.png)

## Pipeline

- The model has 171,653 parameters (169,887 trainable, 1,766 non-trainable), matching the notebook's `model.summary()`; a test checks the counts.
- The train / validation split is made once, up front, and written to `split.json`.
- File names are `<scan>_slice_<n>.png`; by default whole scans are kept on one side of the split (`--no-group-by-scan` turns this off).
- Augmentation is applied to training images only.
- Training uses a mask-only view of the network (the edge branch stays in the graph because it feeds the decoder).
- Metrics (Dice, IoU, precision, recall, F1, specificity) are computed from thresholded predictions.
- Images are read as RGB, resized to 256 x 256 and scaled to [0, 1]; masks use nearest-neighbour resizing.
- Research code only; not validated for clinical use.

## Repository structure

```
.
├── src/carotid_seg/     # model, losses, data, training, evaluation
├── scripts/             # train.py, evaluate.py, predict.py
├── tests/               # unit tests (synthetic data; no dataset needed)
├── notebooks/           # original experiment notebook
├── docs/figures/        # architecture figures
├── results/figures/     # dataset figures from the notebook
└── data/README.md       # dataset instructions
```

Development: `pip install -e ".[dev]"`, then `ruff check .` and `pytest`.

## License

MIT. See [LICENSE](LICENSE). The dataset has its own license; check it before reuse.
