"""flair physics: protocol-predicted shift of wmh-vs-wm log contrast between two scans

signal: |1 - 2 exp(-ti/t1) + exp(-tr/t1)| exp(-te/t2), hajnal 1992
gain and proton density cancel in every ratio
wm t1/t2: rooney 2007, whittall 1997, wansapura 1999, bojorquez 2017 (supplement s2)
wmh t1: munoz maniega 2015, thrippleton 2019
wmh t2: effective in-sample value, not the literature 150 ms. the eroded lesion core behaves like wm
wmh t2 is fitted on the study cohort. other values change the directional calls (supplement s7)
"""

import numpy as np

from wmh_qc.containers import ScanMeta

WMH_T2_EFF_MS = 83.0
# (t1_ms, t2_ms) per tissue
RELAX_1P5T = {"wm": (650.0, 75.0), "wmh": (1000.0, WMH_T2_EFF_MS)}
RELAX_3T = {"wm": (832.0, 75.0), "wmh": (1100.0, WMH_T2_EFF_MS)}
FIELD_3T_MIN_T = 2.0
# relaxometry prior multipliers on t1 and t2. nominal first
PRIOR_MULTS = (1.0, 0.85, 1.15)


def relaxometry(field_t: float) -> dict[str, tuple[float, float]]:
    return RELAX_3T if field_t >= FIELD_3T_MIN_T else RELAX_1P5T


def flair_signal(relax: tuple[float, float], meta: ScanMeta, prior_mult: float) -> float:
    """relative ir-tse signal of one tissue, gain-free"""
    t1_ms = relax[0] * prior_mult
    t2_ms = relax[1] * prior_mult
    inversion = abs(1.0 - 2.0 * np.exp(-meta.ti_ms / t1_ms) + np.exp(-meta.tr_ms / t1_ms))
    return inversion * np.exp(-meta.te_ms / t2_ms)


def contrast_shift(meta1: ScanMeta, meta2: ScanMeta, mode1: float, mode2: float,
                   prior_mult: float) -> float:
    """predicted wmh-vs-wm log-contrast shift of scan 2 vs scan 1 in mode-normalised space"""
    relax1 = relaxometry(meta1.field_t)
    relax2 = relaxometry(meta2.field_t)
    wm1 = flair_signal(relax1["wm"], meta1, prior_mult)
    wm2 = flair_signal(relax2["wm"], meta2, prior_mult)
    wmh1 = flair_signal(relax1["wmh"], meta1, prior_mult)
    wmh2 = flair_signal(relax2["wmh"], meta2, prior_mult)
    gain = mode2 / mode1
    wmh_scale = gain * (wmh2 / wmh1) / (wm2 / wm1)
    # two logs, not one log of the ratio: one log gives other last digits of mu than the paper
    return float(np.log(wmh_scale) - np.log(gain))


def contrast_shifts(meta1: ScanMeta, meta2: ScanMeta, mode1: float,
                    mode2: float) -> tuple[float, ...]:
    """contrast shift at every prior multiplier, nominal first. metadata must be complete"""
    return tuple(contrast_shift(meta1, meta2, mode1, mode2, mult) for mult in PRIOR_MULTS)
