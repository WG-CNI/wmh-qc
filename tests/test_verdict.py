"""flags and verdict rule"""

from wmh_qc.change import Remeasurement
from wmh_qc.verdict import (
    DECREASE,
    FLAG_BIDIRECTIONAL,
    FLAG_COHERENT_DIMMING,
    FLAG_CONTRADICTION,
    INCREASE,
    NOT_RESOLVED,
    contradicts,
    qc_flags,
    verdict,
)


def test_contradiction_needs_opposite_large_values() -> None:
    """opposite signs, both at least 0.5 ml"""
    assert contradicts(-0.81, 2.0)
    assert not contradicts(-0.49, 2.0)
    assert not contradicts(1.0, 2.0)


def test_gate_flags_only_across_scanners() -> None:
    """bidirectional and coherent dimming are ignored on same-scanner pairs. contradiction is not"""
    remeasured = Remeasurement(3.0, 2.0, 1.0, True)
    assert qc_flags(remeasured, True, -1.0) == (FLAG_BIDIRECTIONAL, FLAG_COHERENT_DIMMING, FLAG_CONTRADICTION)
    assert qc_flags(remeasured, False, -1.0) == (FLAG_CONTRADICTION,)


def test_verdict_rule() -> None:
    """flags and zero-spanning ranges block the call, otherwise the sign of dv"""
    assert verdict(5.0, (4.0, 6.0), (FLAG_CONTRADICTION,)) == NOT_RESOLVED
    assert verdict(0.2, (-0.1, 0.5), ()) == NOT_RESOLVED
    assert verdict(5.0, (4.0, 6.0), ()) == INCREASE
    assert verdict(-5.0, (-6.0, -4.0), ()) == DECREASE
