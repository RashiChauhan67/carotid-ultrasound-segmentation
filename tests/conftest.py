import cv2
import numpy as np
import pytest


@pytest.fixture
def carotid_root(tmp_path):
    """12 synthetic slices from 4 scans (3 slices each), with blob masks."""
    images = tmp_path / "data" / "US images"
    masks = tmp_path / "data" / "Expert mask images"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for scan in ("20220101A", "20220102B", "20220103C", "20220104D"):
        for k in range(3):
            name = f"{scan}_slice_{k}.png"
            cv2.imwrite(str(images / name), rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
            mask = np.zeros((48, 64), dtype=np.uint8)
            cv2.circle(mask, (20 + 4 * k, 24), 8, 255, -1)
            cv2.imwrite(str(masks / name), mask)
    return tmp_path / "data"
