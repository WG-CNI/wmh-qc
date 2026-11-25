"""pair statistics: calibration, blur, controls and warp on synthetic heads"""

import dataclasses
from collections.abc import Callable

import numpy as np
import pytest

from wmh_qc.constants import DEFAULT_BLUR_RATIO
from wmh_qc.containers import Scan, ScanMeta
from wmh_qc.harmonisation import (
    apply_warp,
    split_controls,
    tissue_shifts,
)
from wmh_qc.loading import Grid
from wmh_qc.pair_stats import (
    DEFAULT_EMP_NOISE,
    DEFAULT_OFFSET,
    NOISE_FLOOR,
    blur_ratio,
    build_pair,
)
from wmh_qc.tissue import build_scan

ZOOMS = (1.0, 1.0, 1.0)
HALF_MM = (0.5, 0.5, 0.5)


@pytest.fixture
def make_head(volume_shape: tuple[int, int, int], same_meta: ScanMeta) -> Callable[[int, float], Scan]:
    """factory: synthetic head with cortex shell, wm, ventricle, thalamus, one lesion, multiplicative noise"""

    def build(seed: int, gain: float) -> Scan:
        labels = np.zeros(volume_shape, dtype=np.int32)
        labels[2:38, 2:38, 2:38] = 3
        labels[7:33, 7:33, 7:33] = 2
        labels[17:23, 12:28, 17:23] = 4
        labels[27:32, 12:28, 27:32] = 10
        intensity = {0: 0.0, 2: 100.0, 3: 130.0, 4: 20.0, 10: 115.0}
        flair = np.zeros(volume_shape, dtype=np.float32)
        for label, value in intensity.items():
            flair[labels == label] = value
        mask = np.zeros(volume_shape, dtype=bool)
        mask[10:14, 10:14, 10:14] = True
        flair[mask] = 200.0
        noise = np.random.default_rng(seed).normal(1.0, 0.03, volume_shape)
        flair = (gain * flair * noise).astype(np.float32)
        return build_scan(flair, labels, mask, same_meta, Grid(volume_shape, np.eye(4), ZOOMS))

    return build


def test_identical_scans_have_no_offset_and_unit_blur(make_head: Callable[[int, float], Scan]) -> None:
    scan = make_head(0, 1.0)
    pair = build_pair(scan, scan, ZOOMS)
    assert pair.offset == 0.0 and pair.offset_valid
    assert pair.emp_noise == pytest.approx(NOISE_FLOOR)
    assert pair.blur_ratio == 1.0 and pair.blur_valid
    assert pair.span_valid


def test_gain_change_is_absorbed_by_mode_normalisation(make_head: Callable[[int, float], Scan]) -> None:
    pair = build_pair(make_head(0, 1.0), make_head(1, 1.7), ZOOMS)
    assert pair.offset_valid and abs(pair.offset) < 0.01
    assert 0.01 < pair.emp_noise < 0.1
    assert 0.8 < pair.blur_ratio < 1.25
    assert np.isfinite(pair.tissue_shift_span)
    assert set(np.unique(pair.confidence).tolist()) <= {1.0, 0.5, 0.25}
    assert (pair.dist_from_m1_mm[pair.m1] == 0).all()
    assert pair.dist_from_m1_mm.dtype == np.float32


def test_lesion_free_overlap_too_small_gives_default_calibration(
    volume_shape: tuple[int,
    int,
    int],
    make_head: Callable[[int,
    float],
    Scan],
) -> None:
    earlier = make_head(0, 1.0)
    later = dataclasses.replace(make_head(1, 1.0), mask=np.ones(volume_shape, dtype=bool))
    pair = build_pair(earlier, later, ZOOMS)
    assert not pair.offset_valid
    assert pair.emp_noise == DEFAULT_EMP_NOISE and pair.offset == DEFAULT_OFFSET


def test_unmeasurable_blur_is_neutral(make_head: Callable[[int, float], Scan]) -> None:
    """no lesion edge to measure: ratio 1, marked invalid, so it never counts as a sharpness change"""
    earlier = dataclasses.replace(make_head(0, 1.0), mask=np.zeros_like(make_head(0, 1.0).mask))
    assert blur_ratio(earlier, make_head(1, 1.0), ZOOMS) == (DEFAULT_BLUR_RATIO, False)


def test_split_controls_is_a_disjoint_checkerboard() -> None:
    """large controls split on even i+j+k, small ones are used whole"""
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[2:12, 2:12, 2:12] = True
    small = np.zeros_like(mask)
    small[0, 0, 0:5] = True
    fit, held_out = split_controls({"stable_wm": mask, "vent": small}, ZOOMS)
    assert not (fit["stable_wm"] & held_out["stable_wm"]).any()
    np.testing.assert_array_equal(fit["stable_wm"] | held_out["stable_wm"], mask)
    i, j, k = np.nonzero(fit["stable_wm"])
    assert ((i + j + k) % 2 == 0).all()
    assert fit["vent"] is small and held_out["vent"] is small


def test_identity_warp_keeps_intensities_and_extrapolates_linearly() -> None:
    knots = np.linspace(10.0, 100.0, 10, dtype=np.float32)
    fl2 = np.array([0.0, 5.0, 50.0, 150.0], dtype=np.float32)
    np.testing.assert_allclose(apply_warp(fl2, knots, knots), [0.0, 5.0, 50.0, 150.0])


def test_one_tissue_does_not_validate_the_span() -> None:
    """span of a single tissue is 0 by construction: no evidence that the tissues shift alike"""
    image = np.full((10, 10, 10), 100.0)
    controls = {name: np.zeros(image.shape, bool) for name in ("stable_wm", "cortex", "deep_gm", "vent")}
    controls["stable_wm"][:] = True
    assert not tissue_shifts(image, image, image, controls, ZOOMS).valid


def test_empty_baseline_mask_is_infinitely_far(make_head: Callable[[int, float], Scan]) -> None:
    scan = make_head(0, 1.0)
    empty = dataclasses.replace(scan, mask=np.zeros_like(scan.mask))
    assert np.isinf(build_pair(empty, scan, ZOOMS).dist_from_m1_mm).all()
