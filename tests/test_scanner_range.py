"""scanner range geometry and bounds"""

import dataclasses
from collections.abc import Callable

import numpy as np
import pytest
from scipy.ndimage import binary_dilation

from wmh_qc.assess import assess_pair
from wmh_qc.containers import PairData
from wmh_qc.scanner_range import dilate_mm, l1_ball, scanner_range, smoothness


def test_isotropic_dilation_matches_iterated_cross() -> None:
    """isotropic grid: identical to the frozen 6-neighbour dilation with round(2 mm / zoom) iterations"""
    rng = np.random.default_rng(0)
    mask = rng.random((30, 30, 30)) > 0.995
    for zoom in (0.5, 0.43, 1.0):
        iterations = max(1, round(2.0 / zoom))
        frozen = binary_dilation(mask, iterations=iterations)
        assert np.array_equal(dilate_mm(mask, 2.0, (zoom, zoom, zoom)), frozen)


def test_anisotropic_radius_is_per_axis() -> None:
    """thick slices get a one-voxel through-plane radius, not the in-plane count"""
    structure = l1_ball((5, 5, 1))
    assert structure.shape == (11, 11, 3)
    assert structure[5, 5, 0] and structure[5, 5, 2]
    assert not structure[5, 0, 0]


def test_same_scanner_range_brackets_dv(make_pair: Callable[..., PairData]) -> None:
    """same protocol, no blur: zero scanner effect, range = dv +- noise"""
    pair = make_pair()
    domain = pair.valid_zone & ~pair.cleanup_zone
    result = scanner_range(pair, domain, (0.0, 0.0, 0.0))
    expected_dv = float(pair.m2.sum() - pair.m1.sum()) / 1000.0
    assert np.isclose(result.dv_ml, expected_dv)
    assert result.scanner_ml == (0.0, 0.0)
    assert result.blur_ml == 0.0
    assert result.lo_ml <= result.dv_ml <= result.hi_ml
    assert np.isclose(result.hi_ml - result.dv_ml, result.dv_ml - result.lo_ml)


@pytest.mark.parametrize("empty_scan", ["m1", "m2"])
def test_one_empty_mask_gives_finite_range(
    empty_scan: str,
    volume_shape: tuple[int,
    int,
    int],
    make_pair: Callable[...,
    PairData],
) -> None:
    """lesion at one scan only: the empty scan adds no effect, the range stays finite around dv"""
    pair = make_pair(**{empty_scan: np.zeros(volume_shape, bool)})
    result = assess_pair(pair)
    assert np.isfinite([result.range_lo_ml, result.range_hi_ml]).all()
    assert result.range_lo_ml <= result.dv_ml <= result.range_hi_ml


def test_lesions_only_outside_the_domain_add_nothing(make_pair: Callable[..., PairData]) -> None:
    """every lesion in excluded tissue: no crash, no scanner effect, range = dv +- 0"""
    pair = make_pair()
    domain = np.zeros_like(pair.valid_zone)
    result = scanner_range(pair, domain, (0.05, 0.0, 0.1))
    assert result.scanner_ml == (0.0, 0.0)
    assert result.lo_ml == result.dv_ml == result.hi_ml


@pytest.mark.parametrize("zooms_mm", [(0.5, 0.5, 0.5), (1.0, 1.0, 1.0), (0.5, 0.5, 3.0)])
def test_white_noise_is_one_voxel_per_sample(zooms_mm: tuple[float, float, float]) -> None:
    """uncorrelated noise: about one voxel per independent sample on every grid"""
    image = np.random.default_rng(0).normal(0.0, 1.0, (30, 30, 30)).astype(np.float32)
    assert 1.0 <= smoothness(image, np.ones(image.shape, dtype=bool), zooms_mm) < 1.5


def test_lesions_darker_than_white_matter_are_refused(make_pair: Callable[..., PairData]) -> None:
    """a mask that does not belong to its flair: the contrast weights would divide by a non-positive gap"""
    pair = make_pair()
    dark = dataclasses.replace(pair, x1=np.zeros_like(pair.x1))
    with pytest.raises(ValueError, match="not brighter"):
        scanner_range(dark, dark.valid_zone & ~dark.cleanup_zone, (0.05, 0.0, 0.1))
