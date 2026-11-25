"""independent remeasurement of wmh change from the two images, plus its warning signals

same-scanner and resolution-only pairs: binary mask difference, contrast is preserved
all other pairs: cluster-mean change of the offset-corrected log images
a cluster counts only when its mean change clears the model/noise band
per-voxel residuals are noise, the cluster mean is not
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label
from scipy.ndimage import mean as label_mean

from wmh_qc.constants import MM3_PER_ML
from wmh_qc.containers import PairData, Zooms
from wmh_qc.spacing import voxel_count, voxel_ml, voxel_mm3
from wmh_qc.transition import RESOLUTION_CHANGE, SAME_SCANNER_MINOR_DRIFT, Transition

NATIVE_CLASSES = {SAME_SCANNER_MINOR_DRIFT, RESOLUTION_CHANGE}

MIN_CLUSTER_MM3 = 5.0          # 40 voxels at 0.5 mm iso
GROWTH_SHELL_MM = 6.0          # peri-lesional growth search radius
MODEL_FLOOR_LOG = 0.10
NOISE_BAND_MULT = 2.5
CANDIDATE_BAND_FRAC = 0.3      # voxel prefilter, the cluster mean must clear the full band
MIN_LESION_MM3 = 12.5          # 100 voxels at 0.5 mm iso
DEFAULT_LESION_LOG = 0.2
MIN_GATE_LOG = 0.10
GROWTH_GATE_FRAC = 0.4
FIELD_GROWTH_GATE_FRAC = 0.75  # higher field over-detects dim voxels: growth must reach full lesion brightness
SHRINK_GATE_FRAC = 0.4

COHERENT_FRAC_MIN = 0.40       # largest dimming component share of all dimming voxels
COHERENT_ML_MIN = 0.5
BIDIRECTIONAL_MIN_ML = 1.5
BIDIRECTIONAL_MIN_RATIO = 0.5


@dataclass(frozen=True)
class Remeasurement:
    """confidence-weighted growth and shrink in ml, plus the coherent-dimming signal"""

    growth_ml: float
    shrink_ml: float
    net_ml: float
    coherent_dimming: bool


def weighted_ml(mask: np.ndarray, confidence: np.ndarray, volume_mm3: float) -> float:
    # full-volume float32 sum. confidence[mask].sum() adds in another order and changes the last digits
    return float((mask * confidence).sum()) * volume_mm3 / MM3_PER_ML


def keep_clusters_above(candidates: np.ndarray, residual: np.ndarray, band: float, min_voxels: int) -> np.ndarray:
    """candidate components of at least min_voxels whose mean residual exceeds band"""
    labels, count = label(candidates)
    sizes = np.bincount(labels.ravel())
    means = label_mean(residual, labels, index=np.arange(count + 1))
    keep = (sizes >= min_voxels) & (means > band)
    keep[0] = False
    return keep[labels]


def native_change(pair: PairData, domain: np.ndarray) -> Remeasurement:
    """binary difference of the supplied masks inside the domain"""
    volume_mm3 = voxel_mm3(pair.zooms_mm)
    growth = pair.m2 & ~pair.m1 & domain
    shrink = pair.m1 & ~pair.m2 & domain
    growth_ml = weighted_ml(growth, pair.confidence, volume_mm3)
    shrink_ml = weighted_ml(shrink, pair.confidence, volume_mm3)
    return Remeasurement(growth_ml, shrink_ml, growth_ml - shrink_ml, False)


def coherent_dimming(dimming: np.ndarray, zooms_mm: Zooms) -> bool:
    """true when one connected component dominates the lesion dimming and is large"""
    total = int(dimming.sum())
    if total < voxel_count(MIN_CLUSTER_MM3, zooms_mm):
        return False
    labels, _ = label(dimming)
    largest = int(np.bincount(labels.ravel())[1:].max())
    largest_ml = largest * voxel_ml(zooms_mm)
    return largest / total >= COHERENT_FRAC_MIN and largest_ml >= COHERENT_ML_MIN


def cluster_change(pair: PairData, domain: np.ndarray, growth_gate_frac: float) -> Remeasurement:
    """cluster-mean change of the offset-corrected log images"""
    volume_mm3 = voxel_mm3(pair.zooms_mm)
    noise = float(pair.emp_noise)
    band = max(MODEL_FLOOR_LOG, NOISE_BAND_MULT * noise)
    # python float offset keeps the residual float32 (nep 50). the gates below compare in float32 too
    residual = pair.x2 - pair.x1 - float(pair.offset)
    lesion = pair.m1 & domain
    level = DEFAULT_LESION_LOG    # tiny lesion, its median is garbage
    if lesion.sum() > voxel_count(MIN_LESION_MM3, pair.zooms_mm):
        level = float(np.median(pair.x1[lesion]))
    growth_gate = max(growth_gate_frac * level, MIN_GATE_LOG)
    shrink_gate = max(SHRINK_GATE_FRAC * level, MIN_GATE_LOG)

    min_cluster = voxel_count(MIN_CLUSTER_MM3, pair.zooms_mm)
    dimmed = lesion & (pair.x2 < shrink_gate)
    shrink = keep_clusters_above(dimmed & (residual < -CANDIDATE_BAND_FRAC * band), -residual, band, min_cluster)
    shell = (pair.dist_from_m1_mm <= GROWTH_SHELL_MM) & ~pair.m1 & domain
    brightened = shell & (residual > CANDIDATE_BAND_FRAC * band) & (pair.x2 > growth_gate)
    growth = keep_clusters_above(brightened, residual, band, min_cluster)

    growth_ml = weighted_ml(growth, pair.confidence, volume_mm3)
    shrink_ml = weighted_ml(shrink, pair.confidence, volume_mm3)
    coherent = coherent_dimming(dimmed & (residual < -band), pair.zooms_mm)
    return Remeasurement(growth_ml, shrink_ml, growth_ml - shrink_ml, coherent)


def remeasure(pair: PairData, domain: np.ndarray, transition: Transition,
              same_scanner: bool) -> Remeasurement:
    """pick the estimator by transition: native when contrast is preserved, cluster-mean otherwise"""
    if same_scanner or transition.transition_class in NATIVE_CLASSES:
        return native_change(pair, domain)
    field_changed = "field" in transition.changed_fields
    growth_gate_frac = FIELD_GROWTH_GATE_FRAC if field_changed else GROWTH_GATE_FRAC
    return cluster_change(pair, domain, growth_gate_frac)


def is_bidirectional(remeasurement: Remeasurement) -> bool:
    """true when growth and shrink are both large and comparable"""
    smaller = min(remeasurement.growth_ml, remeasurement.shrink_ml)
    larger = max(remeasurement.growth_ml, remeasurement.shrink_ml)
    return smaller >= BIDIRECTIONAL_MIN_ML and smaller >= BIDIRECTIONAL_MIN_RATIO * larger
