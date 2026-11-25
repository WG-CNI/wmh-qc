"""synthseg labels -> wm mode, log image and tissue zones of one scan"""

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label

from wmh_qc.constants import CONNECTIVITY_26
from wmh_qc.containers import Scan, ScanMeta, Tissue, Zooms
from wmh_qc.loading import Grid
from wmh_qc.spacing import voxel_count

WM_LABELS = (2, 41, 77)           # 77 = wmh-synthseg lesion label, counts as wm
CORTEX_LABELS = (3, 42)
VENT_LABELS = (4, 5, 14, 15, 43, 44)
SULCI_LABELS = (24,)              # synthseg "csf" is mostly sulcal csf
CEREB_LABELS = (7, 8, 46, 47)
DEEP_GM_LABELS = (10, 49, 11, 50, 12, 51, 13, 52, 26, 58, 18, 54,
                  17, 53, 28, 60, 16)
CORTEX_RIBBONS = 2                # two hemispheric ribbons. smaller cortex blobs are wm
CEREB_BUFFER_MM = 8.0
PERIVENT_DGM_MM = 5.0
MODE_BINS = 512
MODE_TRIM_PERCENTILES = (5, 95)
LR_CODES = ("L", "R")

# synthseg on the filled flair likes to call periventricular lesions ventricle
# then trust the unfilled segmentation's ventricles
BRIGHT_VENT_GAIN = 1.15           # bright: above this times the wm mode
BRIGHT_VENT_MARGIN = 15.0         # and above mode + this, raw intensity
BRIGHT_VENT_MIN_MM3 = 250.0         # 2000 voxels at 0.5 mm iso
BRIGHT_VENT_MIN_FRACTION = 0.05
BRIGHT_VENT_MIN_EXCESS_MM3 = 125.0  # 1000 voxels at 0.5 mm iso more than in the unfilled ventricles
BRIGHT_VENT_MAX_UNFILLED_RATIO = 0.5


def wm_mode(values: np.ndarray) -> float:
    """centre of the densest bin of the 5-95 percentile trimmed wm intensities"""
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        raise ValueError("no positive wm intensities; check the segmentation")
    # numpy 1 arithmetic of the paper cache: trim bounds in the data dtype, bin edges by float64 linspace
    # cast down. np.histogram(values, MODE_BINS) under numpy 2 puts the edges one ulp off
    lo, hi = np.percentile(values, MODE_TRIM_PERCENTILES).astype(values.dtype)
    values = values[(values >= lo) & (values <= hi)]
    edges = np.linspace(values.min().item(), values.max().item(), MODE_BINS + 1).astype(values.dtype)
    counts, _ = np.histogram(values, bins=edges)
    peak = np.argmax(counts)
    return float(0.5 * (edges[peak] + edges[peak + 1]))


def check_labels(labels: np.ndarray) -> None:
    if not np.isin(labels, CORTEX_LABELS).any() or not np.isin(labels, WM_LABELS).any():
        raise ValueError(f"label map has no cortex {CORTEX_LABELS} or no wm {WM_LABELS} labels; "
                         "run plain synthseg, without --parc")


def white_matter(labels: np.ndarray) -> np.ndarray:
    """wm labels plus every cortex component except the two largest"""
    cortex = np.isin(labels, CORTEX_LABELS)
    components, _ = label(cortex, structure=CONNECTIVITY_26)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    ribbons = np.isin(components, np.argsort(sizes)[-CORTEX_RIBBONS:])
    return np.isin(labels, WM_LABELS) | (cortex & ~ribbons)


def intervent_gap(vent: np.ndarray, brain: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """non-ventricle brain between the outermost ventricle voxels of each left-right row"""
    axis_codes = nib.aff2axcodes(affine)
    lr_axis = next(axis for axis, code in enumerate(axis_codes) if code in LR_CODES)
    seen_before = np.logical_or.accumulate(vent, axis=lr_axis)
    seen_after = np.flip(np.logical_or.accumulate(np.flip(vent, lr_axis), axis=lr_axis), lr_axis)
    return seen_before & seen_after & brain & ~vent


def ventricles(labels: np.ndarray, unfilled_labels: np.ndarray | None,
               brain_flair: np.ndarray, mode: float, zooms_mm: Zooms) -> np.ndarray:
    """ventricles of the filled segmentation, or of the unfilled one when those swallowed lesions"""
    vent = np.isin(labels, VENT_LABELS)
    if unfilled_labels is None:
        return vent
    bright = brain_flair > max(mode * BRIGHT_VENT_GAIN, mode + BRIGHT_VENT_MARGIN)
    unfilled_vent = np.isin(unfilled_labels, VENT_LABELS)
    n_bright = int((vent & bright).sum())
    n_unfilled_bright = int((unfilled_vent & bright).sum())
    many_bright = n_bright >= voxel_count(BRIGHT_VENT_MIN_MM3, zooms_mm)
    bright_share = n_bright / max(int(vent.sum()), 1) >= BRIGHT_VENT_MIN_FRACTION
    beyond_unfilled = n_bright - n_unfilled_bright >= voxel_count(BRIGHT_VENT_MIN_EXCESS_MM3, zooms_mm)
    unfilled_dim = n_unfilled_bright <= BRIGHT_VENT_MAX_UNFILLED_RATIO * n_bright
    swallowed = many_bright and bright_share and beyond_unfilled and unfilled_dim
    return unfilled_vent if swallowed else vent


def log_image(flair: np.ndarray, mode: float) -> np.ndarray:
    out = np.zeros_like(flair, dtype=np.float32)
    positive = flair > 0
    # float32 like the paper cache, written under numpy 1 promotion rules
    out[positive] = np.log(flair[positive]) - np.float32(np.log(mode))
    return out


def build_scan(flair: np.ndarray, labels: np.ndarray, mask: np.ndarray, meta: ScanMeta,
               grid: Grid, *, unfilled_labels: np.ndarray | None = None) -> Scan:
    """per-scan quantities from a prepared flair, its filled synthseg and mask. unfilled_labels feeds the vent check"""
    check_labels(labels)
    if unfilled_labels is not None:
        check_labels(unfilled_labels)
    brain = labels > 0
    brain_flair = flair * brain
    wm = white_matter(labels)
    mode = wm_mode(brain_flair[wm & (brain_flair > 0)])
    vent = ventricles(labels, unfilled_labels, brain_flair, mode, grid.zooms_mm)
    sulci = np.isin(labels, SULCI_LABELS)
    deep_gm = np.isin(labels, DEEP_GM_LABELS)
    cereb = np.isin(labels, CEREB_LABELS)

    vent_dist = distance_transform_edt(~vent, sampling=grid.zooms_mm)
    perivent_dgm = deep_gm & (vent_dist < PERIVENT_DGM_MM)
    del vent_dist
    cereb_buffer = (distance_transform_edt(~cereb, sampling=grid.zooms_mm) < CEREB_BUFFER_MM) & brain
    wm_dilated = binary_dilation(wm, structure=CONNECTIVITY_26) & brain & ~vent & ~sulci

    cleanup_zone = (cereb_buffer | vent | intervent_gap(vent, brain, grid.affine)
                    | sulci | (deep_gm & ~perivent_dgm))
    tissue = Tissue(wm=wm, vent=vent, cortex=np.isin(labels, CORTEX_LABELS),
                    deep_gm=deep_gm, perivent_dgm=perivent_dgm)
    return Scan(log_image=log_image(brain_flair, mode), mode=mode, mask=mask,
                valid_zone=wm_dilated | perivent_dgm, cleanup_zone=cleanup_zone,
                meta=meta, flair=brain_flair, tissue=tissue)
