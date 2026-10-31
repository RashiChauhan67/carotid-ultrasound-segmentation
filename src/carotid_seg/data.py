"""Image/mask loading, a scan-aware train/validation split and an augmenting batch loader."""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import keras
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

IMAGE_SIZE = (256, 256)
IMAGE_DIR = "US images"
MASK_DIR = "Expert mask images"
SPLIT_SEED = 42


def find_pairs(root: str | Path) -> tuple[list[str], list[str]]:
    """Match ultrasound images with their expert masks by file name.

    ``root`` contains ``US images/`` and ``Expert mask images/`` (searched recursively for the
    first folders with these names). Raises if an image has no mask or vice versa.
    """
    root = Path(root)
    image_dir = next(root.rglob(IMAGE_DIR), None)
    mask_dir = next(root.rglob(MASK_DIR), None)
    if image_dir is None or mask_dir is None:
        raise FileNotFoundError(f"Could not find '{IMAGE_DIR}' and '{MASK_DIR}' under {root}")
    images = {p.name: p for p in image_dir.glob("*.png")}
    masks = {p.name: p for p in mask_dir.glob("*.png")}
    if set(images) != set(masks):
        missing = sorted(set(images) ^ set(masks))[:5]
        raise ValueError(f"Images and masks do not match; first differences: {missing}")
    names = sorted(images)
    return [str(images[n]) for n in names], [str(masks[n]) for n in names]


def scan_id(path: str | Path) -> str:
    """Scan (patient) identifier: the file name up to ``_slice_`` (slices of one scan share it)."""
    stem = Path(path).stem
    return stem.rsplit("_slice_", 1)[0] if "_slice_" in stem else stem


def split_pairs(
    image_paths: list[str],
    mask_paths: list[str],
    val_fraction: float = 0.2,
    seed: int = SPLIT_SEED,
    group_by_scan: bool = True,
) -> tuple[tuple[list[str], list[str]], tuple[list[str], list[str]]]:
    """Split into train / validation **before** any shuffling or augmentation.

    With ``group_by_scan`` (default) every slice of a scan goes to the same side, so neighbouring
    slices of one scan cannot leak between training and validation.

    Returns:
        ``((train_images, train_masks), (val_images, val_masks))``.
    """
    idx = np.arange(len(image_paths))
    if group_by_scan:
        groups = [scan_id(p) for p in image_paths]
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_idx, val_idx = next(splitter.split(idx, groups=groups))
    else:
        train_idx, val_idx = train_test_split(idx, test_size=val_fraction, random_state=seed)
    pick = lambda paths, ids: [paths[i] for i in sorted(ids)]  # noqa: E731
    return (
        (pick(image_paths, train_idx), pick(mask_paths, train_idx)),
        (pick(image_paths, val_idx), pick(mask_paths, val_idx)),
    )


def load_pair(
    image_path: str, mask_path: str, size: tuple[int, int] = IMAGE_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """Load an RGB image and its mask, resized to ``size`` and scaled to [0, 1].

    Returns ``(image (H, W, 3) float32, mask (H, W) float32)``.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image at {image_path}")
    if mask is None:
        raise ValueError(f"Failed to load mask at {mask_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mask = mask.astype(np.float32) / 255.0
    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return image, mask


def train_transform(size: tuple[int, int] = IMAGE_SIZE) -> A.Compose:
    """Augmentation pipeline from the notebook (flips, affine, elastic, photometric, blur...).

    The notebook's ``RandomCrop`` is the full image size, so it never changes anything for
    256 x 256 inputs; here its size follows ``size``.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.95, 1.05), translate_percent=(0.05, 0.05), rotate=(-15, 15), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.GaussNoise(std_range=(0.001, 0.005), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GridDistortion(p=0.2),
            A.VerticalFlip(p=0.5),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            A.RandomCrop(height=size[0], width=size[1], p=0.3),
            A.MotionBlur(blur_limit=7, p=0.2),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2),
        ]
    )


class SegmentationData(keras.utils.PyDataset):
    """Batches of ``(images, masks)`` with shapes ``(B, H, W, 3)`` and ``(B, H, W, 1)``.

    Augmentation is applied only when ``augment=True`` (training). The notebook also augmented
    its validation images; here the validation data is left untouched.
    """

    def __init__(
        self,
        image_paths: list[str],
        mask_paths: list[str],
        batch_size: int = 8,
        augment: bool = False,
        shuffle: bool = False,
        size: tuple[int, int] = IMAGE_SIZE,
        seed: int = SPLIT_SEED,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_paths, self.mask_paths = list(image_paths), list(mask_paths)
        self.batch_size, self.size, self.shuffle = batch_size, size, shuffle
        self.transform = train_transform(size) if augment else None
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(self.image_paths))
        if shuffle:
            self.rng.shuffle(self.order)

    def __len__(self) -> int:
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index: int):
        ids = self.order[index * self.batch_size : (index + 1) * self.batch_size]
        images, masks = [], []
        for i in ids:
            image, mask = load_pair(self.image_paths[i], self.mask_paths[i], self.size)
            if self.transform is not None:
                out = self.transform(image=image, mask=mask)
                image, mask = out["image"], out["mask"]
            images.append(image)
            masks.append(mask[..., np.newaxis])
        return np.stack(images).astype("float32"), np.stack(masks).astype("float32")

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.order)
