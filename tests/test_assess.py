"""end-to-end core on synthetic pairs"""

import dataclasses
import math
from collections.abc import Callable

import numpy as np
import pytest

from wmh_qc.assess import assess_pair
from wmh_qc.containers import PairData, ScanMeta
from wmh_qc.verdict import INCREASE, NO_CALL, NOT_RESOLVED


def test_same_scanner_growth_is_increase(make_pair: Callable[..., PairData]) -> None:
    """clean same-scanner growth: consensus, native remeasurement, no flags, increase"""
    result = assess_pair(make_pair())
    assert result.same_scanner
    assert result.flags == ()
    assert result.growth_ml > 0
    assert result.verdict == INCREASE


@pytest.mark.parametrize("missing", [{"te_ms": math.nan}, {"field_t": math.nan}])
def test_missing_metadata_reports_dv_only(
    missing: dict[str,
    float],
    same_meta: ScanMeta,
    make_pair: Callable[...,
    PairData],
) -> None:
    """unknown te or field: no call, dv only, no range and no physics"""
    pair = make_pair(meta2=dataclasses.replace(same_meta, **missing))
    result = assess_pair(pair)
    assert result.verdict == NO_CALL
    assert math.isclose(result.dv_ml, float(pair.m2.sum() - pair.m1.sum()) / 1000.0)
    unreported = (result.range_lo_ml, result.range_hi_ml, result.scanner_ml, result.contrast_ml,
                  result.blur_ml, result.noise_sd_ml, result.mu, result.remeasured_net_ml)
    assert all(value is None for value in unreported)
    assert result.flags == ()


def test_no_lesion_at_either_scan_is_not_resolved(
    volume_shape: tuple[int,
    int,
    int],
    make_pair: Callable[...,
    PairData],
) -> None:
    """two empty masks: dv 0, degenerate range at zero, not resolved"""
    empty = np.zeros(volume_shape, bool)
    result = assess_pair(make_pair(m1=empty, m2=empty))
    assert result.dv_ml == 0.0
    assert (result.range_lo_ml, result.range_hi_ml) == (0.0, 0.0)
    assert result.noise_sd_ml == 0.0
    assert result.verdict == NOT_RESOLVED
