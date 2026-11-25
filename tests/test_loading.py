"""nifti loading onto the baseline grid"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from wmh_qc.loading import load_flair, load_labels, load_mask, read_grid

AFFINE = np.diag([0.5, 0.5, 0.5, 1.0])
SHAPE = (4, 4, 4)
RAW_WM = 300.0                    # raw scanner units


def _save(path: Path, data: np.ndarray, affine: np.ndarray) -> Path:
    nib.save(nib.Nifti1Image(data, affine), path)
    return path


def test_grid_and_flair(tmp_path: Path) -> None:
    path = _save(tmp_path / "flair.nii.gz", np.full(SHAPE, RAW_WM, dtype=np.int16), AFFINE)
    grid = read_grid(path)
    assert grid.shape == (4, 4, 4) and grid.zooms_mm == (0.5, 0.5, 0.5)
    assert load_flair(path, grid).dtype == np.float32


def test_off_grid_flair_is_refused(tmp_path: Path) -> None:
    grid = read_grid(_save(tmp_path / "base.nii.gz", np.ones((4, 4, 4), np.float32), AFFINE))
    shifted = AFFINE.copy()
    shifted[0, 3] = 1.0
    path = _save(tmp_path / "later.nii.gz", np.full(SHAPE, RAW_WM, np.float32), shifted)
    with pytest.raises(ValueError, match="grid"):
        load_flair(path, grid)


def test_mask_is_binarised_and_may_be_empty(tmp_path: Path) -> None:
    grid = read_grid(_save(tmp_path / "base.nii.gz", np.ones((4, 4, 4), np.float32), AFFINE))
    data = np.zeros((4, 4, 4), dtype=np.uint8)
    data[1, 1, 1] = 3
    mask = load_mask(_save(tmp_path / "mask.nii.gz", data, AFFINE), grid)
    assert mask.dtype == bool and mask.sum() == 1
    empty = load_mask(_save(tmp_path / "empty.nii.gz", np.zeros((4, 4, 4), np.uint8), AFFINE), grid)
    assert not empty.any()


def test_coarse_labels_are_resampled_nearest(tmp_path: Path) -> None:
    grid = read_grid(_save(tmp_path / "base.nii.gz", np.ones((4, 4, 4), np.float32), AFFINE))
    coarse = np.arange(8, dtype=np.int32).reshape(2, 2, 2)
    labels = load_labels(_save(tmp_path / "seg.nii.gz", coarse, np.diag([1.0, 1.0, 1.0, 1.0])), grid)
    assert labels.shape == (4, 4, 4) and labels.dtype == np.int32
    assert labels[0, 0, 0] == coarse[0, 0, 0] and labels[2, 2, 2] == coarse[1, 1, 1]


@pytest.mark.parametrize("scale", ["unit", "zscore"])
def test_normalised_flair_is_refused(tmp_path: Path, scale: str) -> None:
    """[0, 1] or z-scored intensities break the raw-unit floors"""
    rng = np.random.default_rng(0)
    raw = rng.normal(RAW_WM, 30.0, SHAPE).astype(np.float32)
    flairs = {"unit": raw / raw.max(), "zscore": (raw - raw.mean()) / raw.std()}
    grid = read_grid(_save(tmp_path / "base.nii.gz", raw, AFFINE))
    path = _save(tmp_path / "flair.nii.gz", flairs[scale], AFFINE)
    with pytest.raises(ValueError, match="raw scanner units"):
        load_flair(path, grid)


def test_negative_background_fill_is_accepted(tmp_path: Path) -> None:
    flair = np.full(SHAPE, -778.0, np.float32)
    flair[1:3, 1:3, 1:3] = RAW_WM
    grid = read_grid(_save(tmp_path / "base.nii.gz", flair, AFFINE))
    np.testing.assert_array_equal(load_flair(tmp_path / "base.nii.gz", grid), flair)


@pytest.mark.parametrize(("values", "dtype"), [((0, 255), np.uint8), ((0.0, 1.0), np.float32)])
def test_binary_mask_in_any_encoding_is_accepted(tmp_path: Path, values: tuple[float, float],
                                                  dtype: type) -> None:
    """0/255 and float 0/1 masks load as the same bool mask"""
    grid = read_grid(_save(tmp_path / "base.nii.gz", np.full(SHAPE, RAW_WM, np.float32), AFFINE))
    data = np.full(SHAPE, values[0], dtype=dtype)
    data[1:3, 1:3, 1:3] = values[1]
    mask = load_mask(_save(tmp_path / "mask.nii.gz", data, AFFINE), grid)
    assert mask.sum() == 8


@pytest.mark.parametrize("nonzero", [(0.5, 0.9), (-1.0,)])
def test_probability_or_negative_mask_is_refused(tmp_path: Path, nonzero: tuple[float, ...]) -> None:
    """lst-lpa / bianca probability maps (or a negative label) are not a binary mask"""
    grid = read_grid(_save(tmp_path / "base.nii.gz", np.full(SHAPE, RAW_WM, np.float32), AFFINE))
    data = np.zeros(SHAPE, dtype=np.float32)
    data[1, 1, :len(nonzero)] = nonzero
    with pytest.raises(ValueError, match="binary"):
        load_mask(_save(tmp_path / "prob.nii.gz", data, AFFINE), grid)


def test_four_dimensional_image_is_refused(tmp_path: Path) -> None:
    path = _save(tmp_path / "4d.nii.gz", np.ones((*SHAPE, 1), np.float32), AFFINE)
    with pytest.raises(ValueError, match="3d"):
        read_grid(path)


def test_metre_geometry_is_refused(tmp_path: Path) -> None:
    image = nib.Nifti1Image(np.ones(SHAPE, np.float32), AFFINE / np.array([1000, 1000, 1000, 1]))
    image.header.set_xyzt_units("meter")
    path = tmp_path / "metres.nii.gz"
    nib.save(image, path)
    with pytest.raises(ValueError, match="spatial units"):
        read_grid(path)


def test_nan_flair_is_refused(tmp_path: Path) -> None:
    data = np.full(SHAPE, RAW_WM, np.float32)
    data[0, 0, 0] = np.nan
    path = _save(tmp_path / "nan.nii.gz", data, AFFINE)
    with pytest.raises(ValueError, match="nan"):
        load_flair(path, read_grid(path))


def test_fractional_labels_are_refused(tmp_path: Path) -> None:
    data = np.full(SHAPE, 2.0, np.float32)
    data[0, 0, 0] = 2.5
    path = _save(tmp_path / "labels.nii.gz", data, AFFINE)
    with pytest.raises(ValueError, match="whole nonnegative labels"):
        load_labels(path, read_grid(path))
