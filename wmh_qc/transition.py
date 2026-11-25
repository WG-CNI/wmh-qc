"""scanner-transition class and multi-signal same-scanner consensus for one scan pair"""

import math
from dataclasses import dataclass
from itertools import takewhile

import numpy as np

from wmh_qc.containers import PairData, ScanMeta

TE_DELTA_MS = 4.0
TR_DELTA_MS = 400.0
TI_DELTA_MS = 120.0
FIELD_DELTA_T = 0.4
THICKNESS_DELTA_MM = 0.8
BLUR_LOG_TOL = 0.10            # |log blur_ratio| beyond this = sharpness changed
BLUR_RATIO_FLOOR = 1e-3

# severity: each delta scaled and capped, plus fixed weights for structural changes
TE_SEVERITY_MS, TE_SEVERITY_CAP = 10.0, 2.0
TR_SEVERITY_MS, TR_SEVERITY_CAP = 1000.0, 1.5
TI_SEVERITY_MS, TI_SEVERITY_CAP = 200.0, 1.5
HARDWARE_WEIGHT = 2.0
MODEL_WEIGHT = 1.0
RESOLUTION_WEIGHT = 1.5
SEVERE_SCORE = 5.0
SEVERE_HARDWARE_CHANGES = 4
SEVERE_ANY_CHANGES = 5

OFFSET_NOISE_MULT = 2.0
SPAN_MAX_LOG = 0.15

MISSING_METADATA = "missing_metadata"
SAME_SCANNER_MINOR_DRIFT = "same_scanner_minor_drift"
SEVERE_MISMATCH = "severe_mismatch"
FIELD_OR_VENDOR_CHANGE = "field_or_vendor_change"
RESOLUTION_CHANGE = "resolution_change"
SAME_VENDOR_PARAM_SHIFT = "same_vendor_param_shift"

HARDWARE_FIELDS = {"field", "vendor"}
RESOLUTION_FIELDS = {"thickness", "blur"}


@dataclass(frozen=True)
class Transition:
    """coarse scanner-transition class of a pair"""

    transition_class: str
    changed_fields: tuple[str, ...]
    severity: float | None        # none when metadata are missing


def blur_log(blur_ratio: float) -> float:
    return abs(np.log(max(blur_ratio, BLUR_RATIO_FLOOR)))


def name_key(name: str) -> str:
    return "".join(char for char in name.lower() if char.isalnum())


def names_match(name1: str, name2: str) -> bool:
    """equal name keys. unknown ("") matches anything"""
    key1 = name_key(name1)
    key2 = name_key(name2)
    return not (key1 and key2) or key1 == key2


def vendor_name(manufacturer: str) -> str:
    """first alphanumeric word of the dicom manufacturer. companies rename themselves, the first word stays"""
    return "".join(takewhile(str.isalnum, manufacturer.strip()))


def same_vendor(meta1: ScanMeta, meta2: ScanMeta) -> bool:
    """same first manufacturer word, or the same known model under a renamed company"""
    model1 = name_key(meta1.model)
    model2 = name_key(meta2.model)
    same_known_model = model1 != "" and model1 == model2
    return same_known_model or names_match(vendor_name(meta1.vendor), vendor_name(meta2.vendor))


def changed_fields(meta1: ScanMeta, meta2: ScanMeta, blur_ratio: float) -> tuple[str, ...]:
    """acquisition properties that materially changed. nan never counts as a change"""
    checks = (
        ("TE", abs(meta2.te_ms - meta1.te_ms) >= TE_DELTA_MS),
        ("TR", abs(meta2.tr_ms - meta1.tr_ms) >= TR_DELTA_MS),
        ("TI", abs(meta2.ti_ms - meta1.ti_ms) >= TI_DELTA_MS),
        ("field", abs(meta2.field_t - meta1.field_t) >= FIELD_DELTA_T),
        ("vendor", not same_vendor(meta1, meta2)),
        ("model", not names_match(meta1.model, meta2.model)),
        ("thickness", abs(meta2.slice_thickness_mm - meta1.slice_thickness_mm) >= THICKNESS_DELTA_MM),
        ("blur", blur_log(blur_ratio) > BLUR_LOG_TOL),
    )
    return tuple(name for name, changed in checks if changed)


def severity_score(meta1: ScanMeta, meta2: ScanMeta, changed: tuple[str, ...]) -> float:
    """protocol-stress score from scaled te/tr/ti deltas and structural changes"""
    score = 0.0
    score += min(abs(meta2.te_ms - meta1.te_ms) / TE_SEVERITY_MS, TE_SEVERITY_CAP)
    score += min(abs(meta2.tr_ms - meta1.tr_ms) / TR_SEVERITY_MS, TR_SEVERITY_CAP)
    score += min(abs(meta2.ti_ms - meta1.ti_ms) / TI_SEVERITY_MS, TI_SEVERITY_CAP)
    score += HARDWARE_WEIGHT * bool(HARDWARE_FIELDS & set(changed))
    score += MODEL_WEIGHT * ("model" in changed)
    score += RESOLUTION_WEIGHT * bool(RESOLUTION_FIELDS & set(changed))
    return float(score)


def classify_transition(meta1: ScanMeta, meta2: ScanMeta, blur_ratio: float) -> Transition:
    """transition class, changed fields and severity of a scan pair"""
    required = (meta1.te_ms, meta1.tr_ms, meta1.ti_ms, meta1.field_t,
                meta2.te_ms, meta2.tr_ms, meta2.ti_ms, meta2.field_t)
    if not all(math.isfinite(value) for value in required):
        return Transition(MISSING_METADATA, (), None)
    changed = changed_fields(meta1, meta2, blur_ratio)
    severity = severity_score(meta1, meta2, changed)
    if not changed:
        return Transition(SAME_SCANNER_MINOR_DRIFT, changed, severity)
    hardware = bool(HARDWARE_FIELDS & set(changed))
    resolution_only = set(changed) <= RESOLUTION_FIELDS
    stressed = severity >= SEVERE_SCORE or len(changed) >= SEVERE_HARDWARE_CHANGES
    if len(changed) >= SEVERE_ANY_CHANGES or (hardware and stressed):
        transition_class = SEVERE_MISMATCH
    elif hardware:
        transition_class = FIELD_OR_VENDOR_CHANGE
    elif resolution_only:
        transition_class = RESOLUTION_CHANGE
    else:
        transition_class = SAME_VENDOR_PARAM_SHIFT
    return Transition(transition_class, changed, severity)


def same_scanner_consensus(pair: PairData, transition: Transition, nominal_shift: float) -> bool:
    """true only when metadata, contrast physics, wm offset, sharpness and tissue span all agree"""
    noise = float(pair.emp_noise)
    no_change = transition.transition_class == SAME_SCANNER_MINOR_DRIFT
    small_shift = abs(nominal_shift) < noise
    small_offset = pair.offset_valid and abs(pair.offset) < OFFSET_NOISE_MULT * noise
    sharp_ok = pair.blur_valid and blur_log(pair.blur_ratio) < BLUR_LOG_TOL
    span_ok = pair.span_valid and pair.tissue_shift_span < SPAN_MAX_LOG
    return bool(no_change and small_shift and small_offset and sharp_ok and span_ok)
