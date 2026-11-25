"""shared fixtures: synthetic volumes, metadata and scan pairs for the qc core tests"""

import dataclasses
from collections.abc import Callable

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from wmh_qc.containers import PairData, ScanMeta

VOLUME_SHAPE = (40, 40, 40)
CENTRE = (20, 20, 20)
LESION_LOG = 0.6
NOISE_LOG = 0.03
SAME_META = ScanMeta(te_ms=82.0, tr_ms=8500.0, ti_ms=2440.0, field_t=1.5,
                     vendor="SIEMENS", model="SymphonyTim", slice_thickness_mm=3.0)


def _ball(radius_vox: float) -> np.ndarray:
    grids = np.ogrid[tuple(slice(0, size) for size in VOLUME_SHAPE)]
    squared = sum((grid - centre) ** 2 for grid, centre in zip(grids, CENTRE, strict=True))
    return squared <= radius_vox ** 2


def _log_image(mask: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.normal(0.0, NOISE_LOG, VOLUME_SHAPE) + LESION_LOG * mask
    return image.astype(np.float32)


@pytest.fixture
def volume_shape() -> tuple[int, int, int]:
    return VOLUME_SHAPE


@pytest.fixture
def same_meta() -> ScanMeta:
    """acquisition metadata of a 1.5 t flair, used for both scans unless a test changes it"""
    return SAME_META


@pytest.fixture
def make_pair() -> Callable[..., PairData]:
    """factory: same-scanner pair with a lesion that grows from radius1 to radius2, fields overridable"""

    def build(radius1_vox: float = 5.0, radius2_vox: float = 6.0, **changes: object) -> PairData:
        m1 = _ball(radius1_vox)
        m2 = _ball(radius2_vox)
        inner = np.zeros(VOLUME_SHAPE, bool)
        inner[2:-2, 2:-2, 2:-2] = True
        pair = PairData(
            x1=_log_image(m1, 1), x2=_log_image(m2, 2), m1=m1, m2=m2,
            valid_zone=inner, cleanup_zone=np.zeros(VOLUME_SHAPE, bool),
            dist_from_m1_mm=distance_transform_edt(~m1).astype(np.float32),
            confidence=np.ones(VOLUME_SHAPE, np.float32), zooms_mm=(1.0, 1.0, 1.0),
            meta1=SAME_META, meta2=SAME_META, mode1=400.0, mode2=420.0,
            emp_noise=0.04, offset=0.0, offset_valid=True, blur_ratio=1.0, blur_valid=True,
            tissue_shift_span=0.05, span_valid=True,
        )
        return dataclasses.replace(pair, **changes)

    return build
