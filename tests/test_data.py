import numpy as np
import pytest

from carotid_seg.data import (
    SegmentationData,
    find_pairs,
    load_pair,
    scan_id,
    split_pairs,
    train_transform,
)


def test_scan_id_strips_the_slice_suffix():
    assert scan_id("/x/20220112174810022VAS_slice_1069.png") == "20220112174810022VAS"
    assert scan_id("/x/20220207125629001EM VASCULAR_slice_13.png") == "20220207125629001EM VASCULAR"
    assert scan_id("plain.png") == "plain"


def test_find_pairs_matches_images_and_masks(carotid_root):
    images, masks = find_pairs(carotid_root)
    assert len(images) == len(masks) == 12
    assert [p.rsplit("/", 1)[-1] for p in images] == [p.rsplit("/", 1)[-1] for p in masks]


def test_find_pairs_rejects_a_missing_mask(carotid_root):
    next((carotid_root / "Expert mask images").glob("*.png")).unlink()
    with pytest.raises(ValueError):
        find_pairs(carotid_root)


def test_group_split_keeps_every_scan_on_one_side(carotid_root):
    images, masks = find_pairs(carotid_root)
    (tr_i, tr_m), (va_i, va_m) = split_pairs(images, masks, val_fraction=0.25)

    assert set(map(scan_id, tr_i)).isdisjoint(set(map(scan_id, va_i)))
    assert len(tr_i) + len(va_i) == 12 and len(va_i) == 3
    assert [p.rsplit("/", 1)[-1] for p in va_i] == [p.rsplit("/", 1)[-1] for p in va_m]


def test_random_split_can_be_requested_and_is_seeded(carotid_root):
    images, masks = find_pairs(carotid_root)
    a = split_pairs(images, masks, 0.25, seed=1, group_by_scan=False)
    b = split_pairs(images, masks, 0.25, seed=1, group_by_scan=False)
    assert a == b


def test_load_pair_resizes_and_scales(carotid_root):
    images, masks = find_pairs(carotid_root)
    image, mask = load_pair(images[0], masks[0], size=(32, 32))
    assert image.shape == (32, 32, 3) and mask.shape == (32, 32)
    assert 0.0 <= image.min() and image.max() <= 1.0
    assert set(np.unique(mask)) <= {0.0, 1.0}


def test_batches_have_the_expected_shapes_and_cover_all_images(carotid_root):
    images, masks = find_pairs(carotid_root)
    data = SegmentationData(images, masks, batch_size=5, size=(32, 32))
    assert len(data) == 3
    x, y = data[0]
    assert x.shape == (5, 32, 32, 3) and y.shape == (5, 32, 32, 1)
    assert data[2][0].shape[0] == 2  # last, smaller batch


def test_augmentation_changes_training_batches_but_not_validation_batches(carotid_root):
    images, masks = find_pairs(carotid_root)
    plain = SegmentationData(images, masks, batch_size=12, size=(64, 64))[0][0]
    again = SegmentationData(images, masks, batch_size=12, size=(64, 64))[0][0]
    np.testing.assert_array_equal(plain, again)  # validation-style data is deterministic

    augmented = SegmentationData(images, masks, batch_size=12, size=(64, 64), augment=True)[0][0]
    assert not np.array_equal(plain, augmented)


def test_train_transform_keeps_masks_binary():
    rng = np.random.default_rng(0)
    image = rng.random((64, 64, 3)).astype("float32")
    mask = (rng.random((64, 64)) > 0.7).astype("float32")
    out = train_transform((64, 64))(image=image, mask=mask)
    assert out["image"].shape == (64, 64, 3) and set(np.unique(out["mask"])) <= {0.0, 1.0}
