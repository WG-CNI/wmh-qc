"""qc of a longitudinal series: every consecutive pair plus baseline to every later scan"""

import logging
from dataclasses import dataclass
from itertools import pairwise

from wmh_qc.assess import assess_pair
from wmh_qc.containers import PairResult, PreparedScan, Scan, ScanMeta
from wmh_qc.loading import Grid, load_flair, load_labels, load_mask, read_grid
from wmh_qc.pair_stats import build_pair
from wmh_qc.tissue import build_scan

LOGGER = logging.getLogger(__name__)

MIN_SCANS = 2


@dataclass(frozen=True)
class SeriesPair:
    """qc result of scan `earlier` versus scan `later` (indices in acquisition order)"""

    earlier: int
    later: int
    result: PairResult


def series_pairs(n_scans: int) -> list[tuple[int, int]]:
    consecutive = [(index, index + 1) for index in range(n_scans - 1)]
    baseline = [(0, later) for later in range(2, n_scans)]
    return consecutive + baseline


def load_scan(files: PreparedScan, meta: ScanMeta, grid: Grid) -> Scan:
    unfilled = None if files.unfilled_labels is None else load_labels(files.unfilled_labels, grid)
    return build_scan(load_flair(files.flair, grid), load_labels(files.labels, grid),
                      load_mask(files.mask, grid), meta, grid, unfilled_labels=unfilled)


def assess_series(prepared: list[PreparedScan], metas: list[ScanMeta]) -> list[SeriesPair]:
    """qc of all reported pairs. the first scan defines the grid"""
    if len(prepared) < MIN_SCANS or len(prepared) != len(metas):
        raise ValueError(f"need at least two scans with one metadata source each, "
                         f"got {len(prepared)} scans and {len(metas)} metadata")
    # dates stay out of logs and errors, they identify the patient
    dates = [meta.acquired for meta in metas]
    if None in dates:
        LOGGER.warning("acquisition date unknown for some scans. scan order not checked")
    elif not all(earlier < later for earlier, later in pairwise(dates)):
        raise ValueError("scan acquisition dates are not strictly increasing. give the scans in acquisition order")
    grid = read_grid(prepared[0].flair)
    # every pair has the baseline or the previous scan as its earlier scan: at most three scans in memory
    baseline = load_scan(prepared[0], metas[0], grid)
    previous = baseline
    results: dict[tuple[int, int], PairResult] = {}
    for later in range(1, len(prepared)):
        scan = load_scan(prepared[later], metas[later], grid)
        results[later - 1, later] = assess_pair(build_pair(previous, scan, grid.zooms_mm))
        if later >= 2:
            results[0, later] = assess_pair(build_pair(baseline, scan, grid.zooms_mm))
        previous = scan
    return [SeriesPair(earlier, later, results[earlier, later]) for earlier, later in series_pairs(len(prepared))]
