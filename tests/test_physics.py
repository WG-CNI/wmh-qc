"""physics: contrast shift between protocols"""

import dataclasses

from wmh_qc.containers import ScanMeta
from wmh_qc.physics import PRIOR_MULTS, RELAX_1P5T, RELAX_3T, contrast_shift, contrast_shifts, relaxometry


def test_identical_protocol_has_zero_shift(same_meta: ScanMeta) -> None:
    assert contrast_shifts(same_meta, same_meta, 400.0, 420.0) == (0.0, 0.0, 0.0)


def test_field_selects_relaxometry() -> None:
    """2 t and above use the 3 t table"""
    assert relaxometry(1.5) is RELAX_1P5T
    assert relaxometry(2.0) is RELAX_3T
    assert relaxometry(3.0) is RELAX_3T


def test_nominal_prior_first(same_meta: ScanMeta) -> None:
    later = dataclasses.replace(same_meta, field_t=3.0, te_ms=100.0)
    shifts = contrast_shifts(same_meta, later, 400.0, 420.0)
    assert PRIOR_MULTS[0] == 1.0
    assert shifts[0] == contrast_shift(same_meta, later, 400.0, 420.0, 1.0)
    assert len(set(shifts)) == len(PRIOR_MULTS)
