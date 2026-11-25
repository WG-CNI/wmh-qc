"""nifti loading onto the baseline grid. errors name the file, never its folder: clinics name folders after patients"""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform

from wmh_qc.containers import Zooms

AFFINE_ATOL_MM = 1e-4
AFFINE_RTOL = 1e-5
# the qc floors are raw scanner units (min intensity 1, tail contrast 1, bright-vent margin 15).
# a raw flair brain has p99 well above 100 (wm 100..4000). a [0, 1] or z-scored image sits below 10
BRAIN_PERCENTILE = 99.0
MIN_BRAIN_P99 = 50.0
# nifti without a unit code is millimetres in practice
MM_UNITS = ("mm", "unknown")


@dataclass(frozen=True)
class Grid:
    """voxel grid of the baseline scan. every other image must sit on it"""

    shape: tuple[int, int, int]
    affine: np.ndarray
    zooms_mm: Zooms


def _load_geometry(path: Path) -> nib.spatialimages.SpatialImage:
    """nifti with a 3d grid in mm: positive finite voxel sizes and an invertible affine"""
    image = nib.load(path)
    if len(image.shape) != 3:
        raise ValueError(f"{path.name} must be a 3d image, it has shape {image.shape}")
    zooms = np.array(image.header.get_zooms())
    bad_zooms = not np.isfinite(zooms).all() or (zooms <= 0).any()
    bad_affine = not np.isfinite(image.affine).all() or np.linalg.det(image.affine[:3, :3]) == 0
    if bad_zooms or bad_affine:
        raise ValueError(f"{path.name} has invalid voxel sizes or affine")
    units = image.header.get_xyzt_units()[0]
    if units not in MM_UNITS:
        raise ValueError(f"{path.name} has spatial units {units}; convert the geometry to mm first")
    return image


def read_grid(path: Path) -> Grid:
    image = _load_geometry(path)
    shape = tuple(int(n) for n in image.shape[:3])
    zooms = tuple(float(z) for z in image.header.get_zooms()[:3])
    return Grid(shape, image.affine, zooms)


def on_grid(image: nib.spatialimages.SpatialImage, grid: Grid) -> bool:
    same_shape = tuple(image.shape[:3]) == grid.shape
    same_affine = np.allclose(image.affine, grid.affine,
                              atol=AFFINE_ATOL_MM, rtol=AFFINE_RTOL)
    return same_shape and same_affine


def _load_on_grid(path: Path, grid: Grid) -> nib.spatialimages.SpatialImage:
    image = _load_geometry(path)
    if not on_grid(image, grid):
        raise ValueError(f"{path.name} is not on the baseline grid; register it first")
    return image


def load_flair(path: Path, grid: Grid) -> np.ndarray:
    """load a flair in raw scanner units as float32"""
    flair = _load_on_grid(path, grid).get_fdata().astype(np.float32)
    if not np.isfinite(flair).all():
        raise ValueError(f"{path.name} has nan or infinite intensities")
    brain = flair[flair > 0]
    # brain = voxels above 0, so a constant negative background fill is ignored
    too_dim = brain.size == 0 or np.percentile(brain, BRAIN_PERCENTILE) < MIN_BRAIN_P99
    if too_dim:
        raise ValueError(f"{path.name} is not in raw scanner units (brain p{BRAIN_PERCENTILE:g} "
                         f"below {MIN_BRAIN_P99:g}); use the unnormalised flair")
    return flair


def load_mask(path: Path, grid: Grid) -> np.ndarray:
    """load a supplied binary wmh mask as bool. empty is valid (no lesions at that scan)"""
    data = np.asanyarray(_load_on_grid(path, grid).dataobj)
    values = np.unique(data[data != 0])
    # one positive value: 0/1, 0/255, float 0/1 all fine. lst-lpa or bianca probabilities are not a mask
    if values.size > 1 or not (values > 0).all():
        raise ValueError(f"{path.name} is not a binary mask: non-zero voxels must share one positive value; "
                         "threshold probability maps first")
    return data > 0


def load_labels(path: Path, grid: Grid) -> np.ndarray:
    """load a synthseg label map as int32, nearest-neighbour resampled to the grid"""
    image = _load_geometry(path)
    data = image.get_fdata()
    # a label map resampled with linear interpolation has fractional labels
    whole = np.isfinite(data).all() and (data == np.round(data)).all()
    if not whole or (data < 0).any():
        raise ValueError(f"{path.name} must hold whole nonnegative labels; resample label maps nearest-neighbour")
    labels = data.astype(np.int32)
    if on_grid(image, grid):
        return labels
    grid_to_labels = np.linalg.inv(image.affine) @ grid.affine
    resampled = affine_transform(labels.astype(np.float32), grid_to_labels[:3, :3],
                                 offset=grid_to_labels[:3, 3],
                                 output_shape=grid.shape, order=0)
    return resampled.astype(np.int32)
