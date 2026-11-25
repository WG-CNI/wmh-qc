"""qc of one scan pair: scanner transition, remeasurement, scanner range, flags, verdict"""

from wmh_qc.change import remeasure
from wmh_qc.containers import PairData, PairResult
from wmh_qc.physics import contrast_shifts
from wmh_qc.scanner_range import mask_change_ml, scanner_range
from wmh_qc.transition import MISSING_METADATA, Transition, classify_transition, same_scanner_consensus
from wmh_qc.verdict import NO_CALL, qc_flags, verdict


def no_call_result(pair: PairData, transition: Transition) -> PairResult:
    return PairResult(
        dv_ml=mask_change_ml(pair), range_lo_ml=None, range_hi_ml=None, scanner_ml=None, contrast_ml=None,
        blur_ml=None, noise_sd_ml=None, mu=None, transition_class=transition.transition_class,
        severity=transition.severity, changed_fields=transition.changed_fields, same_scanner=False,
        remeasured_net_ml=None, growth_ml=None, shrink_ml=None, flags=(), verdict=NO_CALL,
    )


def assess_pair(pair: PairData) -> PairResult:
    """full qc result for one earlier/later scan pair"""
    transition = classify_transition(pair.meta1, pair.meta2, pair.blur_ratio)
    if transition.transition_class == MISSING_METADATA:
        return no_call_result(pair, transition)
    mu = contrast_shifts(pair.meta1, pair.meta2, pair.mode1, pair.mode2)
    same_scanner = same_scanner_consensus(pair, transition, mu[0])
    domain = pair.valid_zone & ~pair.cleanup_zone
    remeasured = remeasure(pair, domain, transition, same_scanner)
    scanner = scanner_range(pair, domain, mu)
    flags = qc_flags(remeasured, not same_scanner, scanner.dv_ml)
    return PairResult(
        dv_ml=scanner.dv_ml,
        range_lo_ml=scanner.lo_ml,
        range_hi_ml=scanner.hi_ml,
        scanner_ml=scanner.scanner_ml,
        contrast_ml=scanner.contrast_ml,
        blur_ml=scanner.blur_ml,
        noise_sd_ml=scanner.noise_sd_ml,
        mu=mu,
        transition_class=transition.transition_class,
        severity=transition.severity,
        changed_fields=transition.changed_fields,
        same_scanner=same_scanner,
        remeasured_net_ml=remeasured.net_ml,
        growth_ml=remeasured.growth_ml,
        shrink_ml=remeasured.shrink_ml,
        flags=flags,
        verdict=verdict(scanner.dv_ml, (scanner.lo_ml, scanner.hi_ml), flags),
    )
