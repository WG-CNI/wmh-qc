"""tissue-shift span: how unevenly tissues still differ after the best intensity harmonisation of the later scan

two candidates, both fitted on one checkerboard half of the control tissues and judged on the other half:
raw local affine, and local affine after a monotone quantile warp. the warp wins only when clearly better
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter

from wmh_qc.constants import CONNECTIVITY_26, MAD_TO_SD
from wmh_qc.containers import Scan, Zooms
from wmh_qc.spacing import morphology_box, voxel_count

# sizes are mm^3 and mm, calibrated to the old voxel numbers on the paper's 0.5 mm iso grid (0.125 mm^3 a voxel)

# control tissues in warp knot order. through np.unique return_index the order reaches the last digits
CONTROL_TISSUES = ("stable_wm", "cortex", "deep_gm", "vent")
STABLE_WM, CORTEX, DEEP_GM, VENT = CONTROL_TISSUES
CONTROL_WM_MARGIN_MM = 5.0
CONTROL_CORTEX_MARGIN_MM = 8.0
CONTROL_DGM_MARGIN_MM = 5.0
CONTROL_CORTEX_EROSION_MM = 0.5   # 1 erosion
CONTROL_VENT_EROSION_MM = 1.0     # 2 erosions
SPLIT_MIN_MM3 = 50.0              # 400 voxels
SPLIT_MIN_HALF_MM3 = 12.5         # 100

# local affine intensity fit
TISSUE_WEIGHTS = {STABLE_WM: 1.0, CORTEX: 1.0, DEEP_GM: 0.8, VENT: 0.5}
AFFINE_MIN_MM3 = 625.0            # 5000 voxels
GAIN_RANGE = (0.5, 1.8)
LOCAL_GAIN_SPREAD = 0.35
LOCAL_SIGMA_MM = 18.0
LOCAL_MIN_SUPPORT = 0.05
DIVISION_FLOOR = 1e-6

# harmonisation qc, log units
QC_MIN_MM3 = 12.5                 # 100 voxels
MIN_INTENSITY = 1.0               # floor before log and after correction
MIN_GAIN = 0.2
SHIFT_MAD_FLOOR = 0.05
QC_MAX_Z = 2.5
QC_MAX_SPAN = 0.25                # also the span unit of the candidate score

# lesion tail check
TAIL_MIN_CORE_MM3 = 6.25          # 50 voxels
TAIL_MIN_WM_MM3 = 25.0            # 200
TAIL_QUANTILE = 0.90
MEDIAN_QUANTILE = 0.5             # np.quantile as in the paper run. np.median gives other last digits
TAIL_MED_RANGE = (0.65, 1.50)
TAIL_HI_RANGE = (0.65, 1.60)
TAIL_CONTRAST_FLOOR = 1.0         # raw units, reference lesion-to-wm contrast
TAIL_FAIL_PENALTY = 2.0
TAIL_PENALTY_WEIGHT = 0.5         # median and upper-tail ratio count equally
TAIL_RATIO_FLOOR = 1e-3

# monotone control warp
WARP_QUANTILES = np.concatenate(([0.002, 0.005, 0.01], np.linspace(0.02, 0.98, 49),
                                 [0.99, 0.995, 0.998]))
WARP_MIN_TISSUE_MM3 = 62.5        # 500 voxels
WARP_MIN_MM3 = 625.0              # 5000
WARP_MIN_KNOTS = 8
WARP_SLOPE_RANGE = (0.20, 5.0)
WARP_TAIL_SLOPE_RANGE = (0.25, 4.0)
WARP_MIN_SCORE_GAIN = 0.15

Controls = dict[str, np.ndarray]


@dataclass(frozen=True)
class TissueShifts:
    """spread of the per-tissue median log shifts after correction"""

    span: float                   # largest minus smallest tissue shift
    max_z: float                  # largest |shift| / its mad-sd
    valid: bool                   # true only when every control tissue had enough voxels


@dataclass(frozen=True)
class TailCheck:
    """lesion-to-wm contrast after correction relative to before"""

    ok: bool
    med_ratio: float
    hi_ratio: float


@dataclass(frozen=True)
class Candidate:
    """qc summary of one harmonisation candidate"""

    span: float
    span_valid: bool
    qc_ok: bool
    tail_ok: bool
    score: float                  # lower is better


def harmonisation_controls(earlier: Scan, later: Scan, stable: np.ndarray,
                           lesion_dist: np.ndarray, zooms_mm: Zooms) -> Controls:
    """lesion-free control tissues present in both scans"""
    t1, t2 = earlier.tissue, later.tissue
    cortex = binary_erosion(t1.cortex & t2.cortex, structure=morphology_box(CONTROL_CORTEX_EROSION_MM, zooms_mm))
    vent = binary_erosion(t1.vent & t2.vent, structure=morphology_box(CONTROL_VENT_EROSION_MM, zooms_mm))
    deep_gm = t1.deep_gm & t2.deep_gm & ~t1.perivent_dgm & ~t2.perivent_dgm
    return {
        STABLE_WM: stable & (lesion_dist >= CONTROL_WM_MARGIN_MM),
        CORTEX: cortex & (lesion_dist >= CONTROL_CORTEX_MARGIN_MM),
        VENT: vent,
        DEEP_GM: deep_gm & (lesion_dist >= CONTROL_DGM_MARGIN_MM),
    }


def split_controls(controls: Controls, zooms_mm: Zooms) -> tuple[Controls, Controls]:
    """checkerboard split into fit and eval halves. small controls are used whole for both"""
    shape = controls[STABLE_WM].shape
    even_parity = np.zeros(shape, dtype=bool)
    even_parity[::2, ::2, ::2] = True
    even_parity[::2, 1::2, 1::2] = True
    even_parity[1::2, ::2, 1::2] = True
    even_parity[1::2, 1::2, ::2] = True
    min_voxels = voxel_count(SPLIT_MIN_MM3, zooms_mm)
    min_half = voxel_count(SPLIT_MIN_HALF_MM3, zooms_mm)
    fit, held_out = {}, {}
    for name, mask in controls.items():
        fit[name], held_out[name] = mask, mask
        if mask.sum() < min_voxels:
            continue
        fit_half, eval_half = mask & even_parity, mask & ~even_parity
        if fit_half.sum() >= min_half and eval_half.sum() >= min_half:
            fit[name], held_out[name] = fit_half, eval_half
    return fit, held_out


def local_affine(fl1: np.ndarray, fl2: np.ndarray, controls: Controls,
                 zooms_mm: Zooms) -> tuple[np.ndarray, np.ndarray]:
    """gaussian-local gain and bias maps of fl2 ~ gain * fl1 + bias on the controls"""
    weights = np.zeros(fl1.shape, dtype=np.float32)
    for name, weight in TISSUE_WEIGHTS.items():
        weights[controls[name]] = np.maximum(weights[controls[name]], weight)
    valid = (weights > 0) & (fl1 > 0) & (fl2 > 0)
    if valid.sum() < voxel_count(AFFINE_MIN_MM3, zooms_mm):
        return np.ones(fl1.shape, dtype=np.float32), np.zeros(fl1.shape, dtype=np.float32)
    # global weighted least squares: fallback where local support is thin, and the local gain clamp
    fl1_valid = fl1[valid].astype(np.float64)
    fl2_valid = fl2[valid].astype(np.float64)
    weights_valid = weights[valid].astype(np.float64)
    del valid
    total = float(weights_valid.sum())
    mean_x0 = float((weights_valid * fl1_valid).sum() / total)
    mean_y0 = float((weights_valid * fl2_valid).sum() / total)
    cov = float((weights_valid * (fl1_valid - mean_x0) * (fl2_valid - mean_y0)).sum())
    var = float((weights_valid * (fl1_valid - mean_x0) ** 2).sum())
    gain0 = float(np.clip(cov / max(var, DIVISION_FLOOR), *GAIN_RANGE))
    bias0 = float(mean_y0 - gain0 * mean_x0)
    del fl1_valid, fl2_valid, weights_valid

    sigma_vox = tuple(LOCAL_SIGMA_MM / zoom_mm for zoom_mm in zooms_mm)
    s0 = gaussian_filter(weights, sigma=sigma_vox)
    sx = gaussian_filter(weights * fl1, sigma=sigma_vox)
    sy = gaussian_filter(weights * fl2, sigma=sigma_vox)
    sxx = gaussian_filter(weights * fl1 * fl1, sigma=sigma_vox)
    sxy = gaussian_filter(weights * fl1 * fl2, sigma=sigma_vox)
    mean_x = sx / np.maximum(s0, DIVISION_FLOOR)
    mean_y = sy / np.maximum(s0, DIVISION_FLOOR)
    gain = (sxy - mean_x * sy) / np.maximum(sxx - mean_x * sx, DIVISION_FLOOR)
    gain_lo = max(GAIN_RANGE[0], gain0 - LOCAL_GAIN_SPREAD)
    gain_hi = min(GAIN_RANGE[1], gain0 + LOCAL_GAIN_SPREAD)
    gain = np.clip(gain, gain_lo, gain_hi).astype(np.float32)
    bias = (mean_y - gain * mean_x).astype(np.float32)
    unsupported = s0 < LOCAL_MIN_SUPPORT
    gain[unsupported] = gain0
    bias[unsupported] = bias0
    return gain, bias


def tissue_shifts(fl1: np.ndarray, fl2: np.ndarray, corrected: np.ndarray,
                  controls: Controls, zooms_mm: Zooms) -> TissueShifts:
    """span and max z of the per-tissue median log shifts"""
    min_voxels = voxel_count(QC_MIN_MM3, zooms_mm)
    shifts, z_scores = [], []
    for mask in controls.values():
        inside = mask & (fl1 > 0) & (fl2 > 0)
        if inside.sum() < min_voxels:
            continue
        diff = (np.log(np.maximum(corrected[inside].astype(np.float64), MIN_INTENSITY))
                - np.log(np.maximum(fl1[inside].astype(np.float64), MIN_INTENSITY)))
        median = float(np.median(diff))
        mad = max(float(np.median(np.abs(diff - median))) * MAD_TO_SD, SHIFT_MAD_FLOOR)
        shifts.append(median)
        z_scores.append(abs(median) / mad)
    if not shifts:
        return TissueShifts(span=0.0, max_z=0.0, valid=False)
    # one measured tissue has span 0 by construction, that proves nothing
    return TissueShifts(span=float(max(shifts) - min(shifts)), max_z=float(max(z_scores)),
                        valid=len(shifts) == len(controls))


def lesion_tail(fl1: np.ndarray, corrected: np.ndarray, stable_eval: np.ndarray,
                lesions: np.ndarray, zooms_mm: Zooms) -> TailCheck:
    """whether correction keeps the lesion-to-wm contrast, at the median and in the upper tail"""
    min_core = voxel_count(TAIL_MIN_CORE_MM3, zooms_mm)
    wm = stable_eval & (fl1 > 0)
    core = binary_erosion(lesions, structure=CONNECTIVITY_26)
    if core.sum() < min_core:
        core = lesions
    core = core & (fl1 > 0)
    if wm.sum() < voxel_count(TAIL_MIN_WM_MM3, zooms_mm) or core.sum() < min_core:
        return TailCheck(ok=True, med_ratio=1.0, hi_ratio=1.0)
    ref_med = float(np.median(fl1[core]) - np.median(fl1[wm]))
    corr_med = float(np.median(corrected[core]) - np.median(corrected[wm]))
    ref_hi = float(np.quantile(fl1[core], TAIL_QUANTILE) - np.quantile(fl1[wm], MEDIAN_QUANTILE))
    corr_hi = float(np.quantile(corrected[core], TAIL_QUANTILE)
                    - np.quantile(corrected[wm], MEDIAN_QUANTILE))
    med_ratio = corr_med / max(ref_med, TAIL_CONTRAST_FLOOR)
    hi_ratio = corr_hi / max(ref_hi, TAIL_CONTRAST_FLOOR)
    med_ok = TAIL_MED_RANGE[0] <= med_ratio <= TAIL_MED_RANGE[1]
    hi_ok = TAIL_HI_RANGE[0] <= hi_ratio <= TAIL_HI_RANGE[1]
    return TailCheck(ok=med_ok and hi_ok, med_ratio=med_ratio, hi_ratio=hi_ratio)


def _quantile_knots(values: np.ndarray) -> np.ndarray:
    """monotone knots at WARP_QUANTILES, interpolated on the empirical cdf"""
    cdf = np.arange(1, values.size + 1, dtype=np.float64) / values.size
    return np.maximum.accumulate(np.interp(WARP_QUANTILES, cdf, np.sort(values)))


def monotone_warp(fl1: np.ndarray, fl2: np.ndarray, controls: Controls,
                  zooms_mm: Zooms) -> tuple[np.ndarray, np.ndarray] | None:
    """fl2 -> fl1 quantile-matching knots (source, target) on the controls. none when unstable"""
    min_tissue = voxel_count(WARP_MIN_TISSUE_MM3, zooms_mm)
    src_parts, dst_parts = [], []
    total = 0
    for name in CONTROL_TISSUES:
        valid = controls[name] & (fl1 > 0) & (fl2 > 0)
        n_valid = int(valid.sum())
        if n_valid < min_tissue:
            continue
        dst_parts.append(_quantile_knots(fl1[valid].astype(np.float64)))
        src_parts.append(_quantile_knots(fl2[valid].astype(np.float64)))
        total += n_valid
    if total < voxel_count(WARP_MIN_MM3, zooms_mm):
        return None
    src, first = np.unique(np.concatenate(src_parts), return_index=True)
    dst = np.maximum.accumulate(np.concatenate(dst_parts)[first])
    if src.size < WARP_MIN_KNOTS:
        return None
    slopes = np.diff(dst) / np.maximum(np.diff(src), DIVISION_FLOOR)
    if slopes.min() < WARP_SLOPE_RANGE[0] or slopes.max() > WARP_SLOPE_RANGE[1]:
        return None
    return src.astype(np.float32), dst.astype(np.float32)


def apply_warp(fl2: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """piecewise-linear warp of fl2, linear extrapolation beyond the knots"""
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    positive = fl2 > 0
    values = fl2[positive].astype(np.float64)
    warped = np.interp(values, src, dst)
    lo_slope = np.clip((dst[1] - dst[0]) / max(src[1] - src[0], DIVISION_FLOOR),
                       *WARP_TAIL_SLOPE_RANGE)
    hi_slope = np.clip((dst[-1] - dst[-2]) / max(src[-1] - src[-2], DIVISION_FLOOR),
                       *WARP_TAIL_SLOPE_RANGE)
    below = values < src[0]
    above = values > src[-1]
    warped[below] = dst[0] + (values[below] - src[0]) * lo_slope
    # keep lesion contrast above the control range, do not let it saturate
    warped[above] = dst[-1] + (values[above] - src[-1]) * hi_slope
    out = np.zeros_like(fl2, dtype=np.float32)
    out[positive] = np.maximum(warped, MIN_INTENSITY).astype(np.float32)
    return out


def harmonisation_candidate(fl1: np.ndarray, fl2: np.ndarray, splits: tuple[Controls, Controls],
                            lesions: np.ndarray, zooms_mm: Zooms) -> Candidate:
    """fit the local affine on the fit half, judge it on the held-out half"""
    fit, held_out = splits
    gain, bias = local_affine(fl1, fl2, fit, zooms_mm)
    corrected = np.maximum((fl2 - bias) / np.maximum(gain, MIN_GAIN), MIN_INTENSITY)
    del gain, bias
    shifts = tissue_shifts(fl1, fl2, corrected, held_out, zooms_mm)
    tail = lesion_tail(fl1, corrected, held_out[STABLE_WM], lesions, zooms_mm)
    penalty = TAIL_FAIL_PENALTY
    if tail.ok:
        penalty = (TAIL_PENALTY_WEIGHT * abs(np.log(max(tail.med_ratio, TAIL_RATIO_FLOOR)))
                   + TAIL_PENALTY_WEIGHT * abs(np.log(max(tail.hi_ratio, TAIL_RATIO_FLOOR))))
    return Candidate(span=shifts.span, span_valid=shifts.valid,
                     qc_ok=shifts.max_z <= QC_MAX_Z and shifts.span <= QC_MAX_SPAN, tail_ok=tail.ok,
                     score=float(shifts.max_z + shifts.span / QC_MAX_SPAN + penalty))


def tissue_shift_span(earlier: Scan, later: Scan, stable: np.ndarray,
                      lesion_dist: np.ndarray, zooms_mm: Zooms) -> Candidate:
    """the better of raw-affine and warp-affine harmonisation of the later scan"""
    controls = harmonisation_controls(earlier, later, stable, lesion_dist, zooms_mm)
    splits = split_controls(controls, zooms_mm)
    lesions = earlier.mask | later.mask
    raw = harmonisation_candidate(earlier.flair, later.flair, splits, lesions, zooms_mm)
    knots = monotone_warp(earlier.flair, later.flair, splits[0], zooms_mm)
    if knots is None:
        return raw
    warped = apply_warp(later.flair, *knots)
    warp = harmonisation_candidate(earlier.flair, warped, splits, lesions, zooms_mm)
    better_state = warp.qc_ok and not raw.qc_ok
    better_score = warp.qc_ok == raw.qc_ok and raw.score - warp.score >= WARP_MIN_SCORE_GAIN
    if warp.tail_ok and (better_state or better_score):
        return warp
    return raw
