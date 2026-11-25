"""transition class and same-scanner consensus"""

import dataclasses
import math
from collections.abc import Callable

from wmh_qc.containers import PairData, ScanMeta
from wmh_qc.transition import (
    MISSING_METADATA,
    RESOLUTION_CHANGE,
    SAME_SCANNER_MINOR_DRIFT,
    SEVERE_MISMATCH,
    changed_fields,
    classify_transition,
    same_scanner_consensus,
)


def test_vendor_name_variants_are_not_a_change(same_meta: ScanMeta) -> None:
    """renamed company, new name over the same model, case, padding, unknown: same vendor. another maker is a change"""
    for vendor in (same_meta.vendor.lower() + " healthcare", "Acquirer Corp", " siemens ", ""):
        assert changed_fields(same_meta, dataclasses.replace(same_meta, vendor=vendor), 1.0) == ()
    other = dataclasses.replace(same_meta, vendor="other medical systems", model="other model")
    assert changed_fields(same_meta, other, 1.0) == ("vendor", "model")


def test_published_cohort_vendor_and_model(same_meta: ScanMeta) -> None:
    """SIEMENS vs Siemens Healthineers same vendor. SymphonyTim vs MAGNETOM Altea and Symphony a model change"""
    altea = dataclasses.replace(same_meta, vendor="Siemens Healthineers", model="MAGNETOM Altea")
    symphony = dataclasses.replace(same_meta, model="Symphony")
    assert changed_fields(same_meta, altea, 1.0) == ("model",)
    assert changed_fields(same_meta, symphony, 1.0) == ("model",)


def test_model_punctuation_does_not_count(same_meta: ScanMeta) -> None:
    """underscores, spaces and case in model and vendor names are not changes"""
    prisma = dataclasses.replace(same_meta, model="Prisma_fit")
    signa = dataclasses.replace(same_meta, vendor="GE MEDICAL SYSTEMS", model="Signa  HDxt")
    assert changed_fields(prisma, dataclasses.replace(prisma, model="Prisma fit"), 1.0) == ()
    assert changed_fields(signa, dataclasses.replace(signa, vendor="GE_MEDICAL", model="signa hdxt"), 1.0) == ()


def test_blur_ratio_sets_resolution_class(same_meta: ScanMeta) -> None:
    transition = classify_transition(same_meta, same_meta, 0.6)
    assert transition.transition_class == RESOLUTION_CHANGE
    assert transition.changed_fields == ("blur",)


def test_missing_field_is_missing_metadata(same_meta: ScanMeta) -> None:
    unknown = dataclasses.replace(same_meta, field_t=math.nan)
    transition = classify_transition(same_meta, unknown, 1.0)
    assert transition.transition_class == MISSING_METADATA
    assert transition.severity is None


def test_many_changes_are_severe(same_meta: ScanMeta) -> None:
    """five or more changed properties are severe regardless of hardware"""
    later = dataclasses.replace(same_meta, te_ms=120.0, tr_ms=5000.0, ti_ms=1800.0, model="Avanto",
                                slice_thickness_mm=5.0)
    transition = classify_transition(same_meta, later, 1.0)
    assert transition.transition_class == SEVERE_MISMATCH
    assert len(transition.changed_fields) == 5


def test_consensus_needs_every_signal(make_pair: Callable[..., PairData]) -> None:
    """clean pair passes. unmeasured blur or a predicted contrast shift above noise fails it"""
    pair = make_pair()
    transition = classify_transition(pair.meta1, pair.meta2, pair.blur_ratio)
    assert transition.transition_class == SAME_SCANNER_MINOR_DRIFT
    assert same_scanner_consensus(pair, transition, 0.0)
    assert not same_scanner_consensus(dataclasses.replace(pair, blur_valid=False), transition, 0.0)
    assert not same_scanner_consensus(pair, transition, 0.05)
