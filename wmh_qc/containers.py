"""shared data contract between the input layer and the qc core"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

Zooms = tuple[float, float, float]  # voxel size per axis, mm


@dataclass(frozen=True)
class PreparedScan:
    """prepared files of one scan, all on the baseline grid"""

    flair: Path                   # skull-stripped, registered, n4
    mask: Path                    # supplied wmh mask
    labels: Path                  # synthseg of the lesion-filled flair
    unfilled_labels: Path | None  # robust synthseg of the unfilled flair, optional


@dataclass(frozen=True)
class ScanMeta:
    """acquisition metadata of one flair scan. nan, "" or none when unknown"""

    te_ms: float
    tr_ms: float
    ti_ms: float
    field_t: float
    vendor: str
    model: str
    slice_thickness_mm: float
    acquired: datetime | None = None  # acquisition date and time, only orders the scans


@dataclass(frozen=True)
class Tissue:
    """synthseg tissue masks of one scan that the pair statistics need"""

    wm: np.ndarray               # bool, wm plus small cortex blobs
    vent: np.ndarray             # bool, ventricles
    cortex: np.ndarray           # bool, cortical grey matter
    deep_gm: np.ndarray          # bool, deep grey matter
    perivent_dgm: np.ndarray     # bool, deep gm near the ventricles


@dataclass(frozen=True)
class Scan:
    """one flair scan on the baseline grid with its supplied mask and tissue zones"""

    log_image: np.ndarray        # float32, log(flair) - log(wm mode)
    mode: float                  # wm intensity mode, raw units
    mask: np.ndarray             # bool, supplied wmh mask
    valid_zone: np.ndarray       # bool, wm domain of this scan
    cleanup_zone: np.ndarray     # bool, excluded tissue of this scan
    meta: ScanMeta
    flair: np.ndarray            # float32, raw intensities, 0 outside brain
    tissue: Tissue


@dataclass(frozen=True)
class PairData:
    """everything the qc core needs for one earlier/later scan pair"""

    x1: np.ndarray               # float32 log image, earlier scan
    x2: np.ndarray               # float32 log image, later scan
    m1: np.ndarray               # bool supplied mask, earlier scan
    m2: np.ndarray               # bool supplied mask, later scan
    valid_zone: np.ndarray       # bool, union over the pair
    cleanup_zone: np.ndarray     # bool, union over the pair
    dist_from_m1_mm: np.ndarray  # float32, distance outside m1
    confidence: np.ndarray       # float32 in {1, 0.5, 0.25}
    zooms_mm: Zooms
    meta1: ScanMeta
    meta2: ScanMeta
    mode1: float
    mode2: float
    emp_noise: float             # sd of x2 - x1 in stable wm
    offset: float                # median of x2 - x1 in stable wm
    offset_valid: bool
    blur_ratio: float            # edge gradient later / earlier
    blur_valid: bool
    tissue_shift_span: float
    span_valid: bool


@dataclass(frozen=True)
class PairResult:
    """qc output for one pair, volumes in ml. no-call pairs carry dv, class, fields and verdict, the rest is none"""

    dv_ml: float
    range_lo_ml: float | None
    range_hi_ml: float | None
    scanner_ml: tuple[float, float] | None  # contrast + blur effect, eq. 7 bounds
    contrast_ml: tuple[float, float] | None
    blur_ml: float | None
    noise_sd_ml: float | None
    mu: tuple[float, ...] | None  # contrast shift at prior 1.0, 0.85, 1.15
    transition_class: str
    severity: float | None
    changed_fields: tuple[str, ...]
    same_scanner: bool
    remeasured_net_ml: float | None
    growth_ml: float | None
    shrink_ml: float | None
    flags: tuple[str, ...]
    verdict: str                  # increase | decrease | not_resolved | no_call
