# Data

No data is stored in this repository. Download the dataset yourself and check its license and terms before use.

## Dataset

**Common Carotid Artery Ultrasound Images**: ultrasound images of the common carotid artery with expert segmentation masks. The notebook read a Kaggle copy (slug `carotid-ultrasound-images`) and found **1,100 matched image / mask pairs**.

Layout (the loader searches the data folder recursively for the two sub-folders):

```
carotid-ultrasound-images/
└── Common Carotid Artery Ultrasound Images/
    └── Common Carotid Artery Ultrasound Images/
        ├── US images/            *.png  (RGB ultrasound, about 749 x 709 px)
        └── Expert mask images/   *.png  (grayscale, artery = 255)
```

Image and mask share a file name, for example `202201121748100022VAS_slice_1069.png`. The part before `_slice_` identifies the scan; many slices come from each scan, which is why the split is grouped by scan.

## Preprocessing

- Images are read as RGB, resized to 256 x 256 (bilinear) and scaled to [0, 1].
- Masks are read as grayscale, scaled to [0, 1] and resized with nearest-neighbour interpolation.
- Training images are augmented (flips, affine, elastic and grid distortion, brightness, gamma, noise, blur, CLAHE, sharpen); validation images are not.

## Source

The notebook records only the Kaggle dataset name. Please confirm the exact Kaggle page, the original source of the images, and the license for your copy, and cite the dataset's original publication when you publish results.
