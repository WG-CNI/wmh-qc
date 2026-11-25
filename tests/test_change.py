"""remeasurement estimators and their gates"""

from collections.abc import Callable

import numpy as np

from wmh_qc.change import (
    MIN_CLUSTER_MM3,
    Remeasurement,
    is_bidirectional,
    keep_clusters_above,
    native_change,
)
from wmh_qc.containers import PairData
from wmh_qc.spacing import voxel_count

HALF_MM = (0.5, 0.5, 0.5)


def test_native_change_is_mask_difference(make_pair: Callable[..., PairData]) -> None:
    pair = make_pair()
    domain = pair.valid_zone & ~pair.cleanup_zone
    result = native_change(pair, domain)
    added = int((pair.m2 & ~pair.m1).sum())
    assert np.isclose(result.growth_ml, added / 1000.0)
    assert result.shrink_ml == 0.0
    assert not result.coherent_dimming


def test_small_clusters_are_dropped() -> None:
    candidates = np.zeros((20, 20, 20), bool)
    candidates[2:4, 2:4, 2:4] = True
    candidates[10:15, 10:15, 10:15] = True
    residual = np.ones((20, 20, 20), np.float32)
    kept = keep_clusters_above(candidates, residual, 0.5, voxel_count(MIN_CLUSTER_MM3, HALF_MM))
    assert int(kept.sum()) == 125


def test_bidirectional_needs_both_large_and_comparable() -> None:
    """growth and shrink both >= 1.5 ml and within a factor of two"""
    assert is_bidirectional(Remeasurement(2.0, 1.6, 0.4, False))
    assert not is_bidirectional(Remeasurement(4.0, 1.6, 2.4, False))
    assert not is_bidirectional(Remeasurement(1.4, 1.4, 0.0, False))
