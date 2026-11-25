"""qc flags and the final increase / decrease / not_resolved / no_call verdict of a pair"""

from wmh_qc.change import Remeasurement, is_bidirectional

CONTRADICTION_MIN_ML = 0.5

FLAG_BIDIRECTIONAL = "bidirectional_growth_and_shrink"
FLAG_COHERENT_DIMMING = "coherent_dimming_blocks_growth"
FLAG_CONTRADICTION = "contradiction"

INCREASE = "increase"
DECREASE = "decrease"
NOT_RESOLVED = "not_resolved"
NO_CALL = "no_call"           # missing metadata. assess_pair returns it before any qc


def contradicts(dv_ml: float, remeasured_net_ml: float) -> bool:
    opposite = dv_ml * remeasured_net_ml < 0
    both_large = abs(dv_ml) >= CONTRADICTION_MIN_ML and abs(remeasured_net_ml) >= CONTRADICTION_MIN_ML
    return opposite and both_large


def qc_flags(remeasurement: Remeasurement, cross_scanner: bool, dv_ml: float) -> tuple[str, ...]:
    """flags that block a directional call. the remeasurement gates apply only across scanners"""
    flags = []
    if cross_scanner and is_bidirectional(remeasurement):
        flags.append(FLAG_BIDIRECTIONAL)
    if cross_scanner and remeasurement.coherent_dimming and remeasurement.net_ml > 0:
        flags.append(FLAG_COHERENT_DIMMING)
    if contradicts(dv_ml, remeasurement.net_ml):
        flags.append(FLAG_CONTRADICTION)
    return tuple(flags)


def verdict(dv_ml: float, range_ml: tuple[float, float], flags: tuple[str, ...]) -> str:
    """direction call for a pair with metadata: only unflagged pairs whose range excludes zero get one"""
    lo_ml, hi_ml = range_ml
    if flags or lo_ml <= 0 <= hi_ml:
        return NOT_RESOLVED
    return INCREASE if dv_ml > 0 else DECREASE
