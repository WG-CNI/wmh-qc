"""pair statistics of an earlier and a later scan on the same grid: noise, offset, sharpness, weights, tissue span"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

from wmh_qc.constants import CONNECTIVITY_26, DEFAULT_BLUR_RATIO, MAD_TO_SD
from wmh_qc.containers import PairData, Scan, Zooms
from wmh_qc.harmonisation import tissue_shift_span
from wmh_qc.spacing import voxel_count

# sizes are mm^3 and mm, calibrated to the old voxel numbers on the paper's 0.5 mm iso grid (0.125 mm^3 a voxel)

# stable-wm calibration, log units
STABLE_MARGIN_MM = 3.0            # from lesions, ventricles and the brain edge
STABLE_MIN_CANDIDATES_MM3 = 125.0  # 1000 voxels
STABLE_MIN_STRICT_MM3 = 62.5      # 500
STABLE_MIN_MM3 = 12.5             # 100
STABLE_STRICT_MADS = 2.5
STABLE_LOOSE_MADS = 3.0
STABLE_UPPER_QUANTILE = 0.97
NOISE_FLOOR = 0.01
DEFAULT_EMP_NOISE = 0.042
DEFAULT_OFFSET = 0.0

# lesion-boundary blur
BLUR_MIN_SHELL_MM3 = 1.25         # 10 voxels
GRADIENT_FLOOR = 1e-6

# confidence weights
CONFIDENCE_VENT_MM = 3.0
CONFIDENCE_FACTOR = 0.5


@dataclass(frozen=True)
class StableWm:
    """noise and offset of x2 - x1 in lesion-free wm, and the voxels they come from"""

    emp_noise: float
    offset: float
    valid: bool                   # false: too few stable voxels, noise and offset are the defaults
    mask: np.ndarray              # bool, stable voxels


def _robust_spread(values: np.ndarray) -> tuple[float, float, float]:
    """median, floored mad-sd and upper quantile of one scan's candidate values"""
    median = float(np.median(values))
    sd = max(float(np.median(np.abs(values - median))) * MAD_TO_SD, NOISE_FLOOR)
    return median, sd, float(np.quantile(values, STABLE_UPPER_QUANTILE))


def stable_wm(earlier: Scan, later: Scan, lesion_dist: np.ndarray, vent_dist: np.ndarray,
              zooms_mm: Zooms) -> StableWm:
    """noise and offset of x2 - x1 in wm far from lesions, ventricles and the brain edge"""
    both_brain = (earlier.flair > 0) & (later.flair > 0)
    brain_edge = binary_dilation(both_brain, structure=CONNECTIVITY_26) & ~both_brain
    candidates = (earlier.tissue.wm & later.tissue.wm
                  & (earlier.flair > earlier.mode) & (later.flair > later.mode)
                  & (lesion_dist > STABLE_MARGIN_MM) & (vent_dist > STABLE_MARGIN_MM)
                  & (distance_transform_edt(~brain_edge, sampling=zooms_mm) > STABLE_MARGIN_MM))
    stable = np.zeros(candidates.shape, dtype=bool)
    if candidates.sum() < voxel_count(STABLE_MIN_CANDIDATES_MM3, zooms_mm):
        return StableWm(DEFAULT_EMP_NOISE, DEFAULT_OFFSET, False, stable)

    x1 = earlier.log_image[candidates]
    x2 = later.log_image[candidates]
    med1, sd1, hi1 = _robust_spread(x1)
    med2, sd2, hi2 = _robust_spread(x2)
    keep = ((np.abs(x1 - med1) <= STABLE_STRICT_MADS * sd1)
            & (np.abs(x2 - med2) <= STABLE_STRICT_MADS * sd2)
            & (x1 <= hi1) & (x2 <= hi2))
    if keep.sum() < voxel_count(STABLE_MIN_STRICT_MM3, zooms_mm):
        keep = ((np.abs(x1 - med1) <= STABLE_LOOSE_MADS * sd1)
                & (np.abs(x2 - med2) <= STABLE_LOOSE_MADS * sd2))
    if keep.sum() < voxel_count(STABLE_MIN_MM3, zooms_mm):
        return StableWm(DEFAULT_EMP_NOISE, DEFAULT_OFFSET, False, stable)

    diff = x2[keep] - x1[keep]
    offset = float(np.median(diff))
    emp_noise = max(MAD_TO_SD * float(np.median(np.abs(diff - offset))), NOISE_FLOOR)
    stable[candidates] = keep
    return StableWm(emp_noise, offset, True, stable)


def _gradient_magnitude(volume: np.ndarray) -> np.ndarray:
    """forward-difference gradient magnitude in voxel units"""
    squares = [np.diff(volume, axis=axis, prepend=volume.take([0], axis=axis)) ** 2
               for axis in range(volume.ndim)]
    # sum() starts at int 0, 0 + square is exact for squares >= +0. axis order 0, 1, 2 keeps the paper digits
    return np.sqrt(sum(squares))


def blur_ratio(earlier: Scan, later: Scan, zooms_mm: Zooms) -> tuple[float, bool]:
    """median gradient ratio later/earlier on the earlier lesion boundary, and whether it was measurable"""
    shell = (binary_dilation(earlier.mask, structure=CONNECTIVITY_26)
             & ~binary_erosion(earlier.mask, structure=CONNECTIVITY_26)
             & (earlier.flair > earlier.mode) & (later.flair > later.mode))
    if shell.sum() < voxel_count(BLUR_MIN_SHELL_MM3, zooms_mm):
        return DEFAULT_BLUR_RATIO, False
    grad1 = np.median(_gradient_magnitude(earlier.log_image)[shell])
    grad2 = np.median(_gradient_magnitude(later.log_image)[shell])
    if min(grad1, grad2) <= GRADIENT_FLOOR:
        return DEFAULT_BLUR_RATIO, False
    return float(grad2 / grad1), True


def distance_mm(mask: np.ndarray, zooms_mm: Zooms) -> np.ndarray:
    """distance to the nearest mask voxel. infinite for an empty mask"""
    # scipy gives small junk distances, not infinity, when the input has no zero voxel
    if not mask.any():
        return np.full(mask.shape, np.inf)
    return distance_transform_edt(~mask, sampling=zooms_mm)


def build_pair(earlier: Scan, later: Scan, zooms_mm: Zooms) -> PairData:
    """all qc inputs of one earlier/later pair"""
    lesion_dist = distance_mm(earlier.mask | later.mask, zooms_mm)
    vent_dist = distance_transform_edt(~(earlier.tissue.vent | later.tissue.vent), sampling=zooms_mm)
    stable = stable_wm(earlier, later, lesion_dist, vent_dist, zooms_mm)
    # voxel weights: halved near ventricles outside m1 and again where the wm labels disagree
    confidence = np.ones(earlier.mask.shape, dtype=np.float32)
    confidence[(vent_dist < CONFIDENCE_VENT_MM) & ~earlier.mask] *= CONFIDENCE_FACTOR
    confidence[earlier.tissue.wm ^ later.tissue.wm] *= CONFIDENCE_FACTOR
    del vent_dist
    harmonised = tissue_shift_span(earlier, later, stable.mask, lesion_dist, zooms_mm)
    del lesion_dist
    ratio, ratio_valid = blur_ratio(earlier, later, zooms_mm)
    return PairData(
        x1=earlier.log_image, x2=later.log_image, m1=earlier.mask, m2=later.mask,
        valid_zone=earlier.valid_zone | later.valid_zone,
        cleanup_zone=earlier.cleanup_zone | later.cleanup_zone,
        dist_from_m1_mm=distance_mm(earlier.mask, zooms_mm).astype(np.float32),
        confidence=confidence, zooms_mm=zooms_mm, meta1=earlier.meta, meta2=later.meta,
        mode1=earlier.mode, mode2=later.mode, emp_noise=stable.emp_noise, offset=stable.offset,
        offset_valid=stable.valid, blur_ratio=ratio, blur_valid=ratio_valid,
        tissue_shift_span=harmonised.span, span_valid=harmonised.span_valid)
