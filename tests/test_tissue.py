"""per-scan tissue: wm mode, cortex blobs, ventricle gap, log image, zones"""

import numpy as np
import pytest

from wmh_qc.containers import ScanMeta
from wmh_qc.loading import Grid
from wmh_qc.tissue import build_scan, intervent_gap, log_image, ventricles, white_matter, wm_mode

ONE_MM = (1.0, 1.0, 1.0)
HALF_MM = (0.5, 0.5, 0.5)


def test_wm_mode_finds_the_peak() -> None:
    """mode sits on the dominant peak despite a flat background"""
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(400.0, 10.0, 20000), rng.uniform(1.0, 2000.0, 2000)])
    assert abs(wm_mode(values.astype(np.float32)) - 400.0) < 5.0


def test_wm_mode_refuses_empty_wm() -> None:
    """no wm intensities is an error, not a silent mode of 1"""
    with pytest.raises(ValueError):
        wm_mode(np.zeros(10, dtype=np.float32))


def test_small_cortex_blobs_become_wm() -> None:
    """only the two largest cortex components stay cortex"""
    labels = np.zeros((12, 12, 12), dtype=np.int32)
    labels[0:3] = 3
    labels[9:12] = 42
    labels[6, 6, 6] = 3
    labels[4, 4, 4] = 2
    wm = white_matter(labels)
    assert wm[6, 6, 6] and wm[4, 4, 4]
    assert not wm[0:3].any() and not wm[9:12].any()


def test_intervent_gap_fills_between_ventricles_along_left_right() -> None:
    """gap spans the row between the outer ventricle voxels only"""
    vent = np.zeros((10, 3, 3), dtype=bool)
    vent[2, 1, 1] = vent[7, 1, 1] = True
    vent[5, 0, 0] = True
    gap = intervent_gap(vent, np.ones_like(vent), np.eye(4))
    expected = np.zeros_like(vent)
    expected[3:7, 1, 1] = True
    np.testing.assert_array_equal(gap, expected)


def test_log_image_is_zero_outside_the_brain() -> None:
    flair = np.array([0.0, 100.0, 200.0], dtype=np.float32)
    out = log_image(flair, 100.0)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [0.0, 0.0, np.log(2.0)], rtol=1e-6)


def _head_labels() -> np.ndarray:
    """cortex shell, wm, one ventricle"""
    labels = np.zeros((30, 30, 30), dtype=np.int32)
    labels[2:28, 2:28, 2:28] = 3
    labels[6:24, 6:24, 6:24] = 2
    labels[12:18, 10:20, 12:18] = 4
    return labels


def test_build_scan_zones(same_meta: ScanMeta) -> None:
    """ventricles are cleanup, wm is valid, mode is the wm intensity"""
    labels = _head_labels()
    flair = np.where(labels == 2, 100.0, 130.0).astype(np.float32) * (labels > 0)
    flair[labels == 4] = 20.0
    mask = np.zeros(labels.shape, dtype=bool)
    mask[8:10, 8:10, 8:10] = True
    scan = build_scan(flair, labels, mask, same_meta, Grid(labels.shape, np.eye(4), ONE_MM))
    vent = labels == 4
    assert scan.cleanup_zone[vent].all()
    assert not scan.valid_zone[vent].any()
    assert scan.valid_zone[labels == 2].sum() > 0.9 * (labels == 2).sum() - vent.sum()
    assert abs(scan.mode - 100.0) < 1.0
    assert (scan.log_image[labels == 0] == 0).all()
    assert scan.flair.dtype == np.float32


def test_ventricles_switch_to_unfilled_when_lesions_were_swallowed(volume_shape: tuple[int, int, int]) -> None:
    """many bright filled-ventricle voxels absent from the unfilled ventricles trigger the switch"""
    labels = np.zeros(volume_shape, dtype=np.int32)
    labels[:, :, :20] = 4
    unfilled = np.zeros_like(labels)
    unfilled[:, :, :2] = 4
    flair = np.full(labels.shape, 100.0, dtype=np.float32)
    flair[:, :, 2:20] = 300.0
    np.testing.assert_array_equal(ventricles(labels, unfilled, flair, 100.0, HALF_MM), unfilled == 4)
    np.testing.assert_array_equal(ventricles(labels, None, flair, 100.0, HALF_MM), labels == 4)
    dim = np.full(labels.shape, 100.0, dtype=np.float32)
    np.testing.assert_array_equal(ventricles(labels, unfilled, dim, 100.0, HALF_MM), labels == 4)


@pytest.mark.parametrize("parcellated", ["labels", "unfilled"])
def test_parcellated_labels_are_refused(same_meta: ScanMeta, parcellated: str) -> None:
    """synthseg --parc or aparc+aseg: cortex split into 1000s/2000s, no label 3 or 42"""
    labels = _head_labels()
    parc = np.where(labels == 3, 1028, labels)
    flair = np.where(labels > 0, 100.0, 0.0).astype(np.float32)
    grid = Grid(labels.shape, np.eye(4), ONE_MM)
    inputs = {"labels": labels, "unfilled": labels, parcellated: parc}
    with pytest.raises(ValueError, match="parc"):
        build_scan(flair, inputs["labels"], np.zeros(labels.shape, bool), same_meta, grid,
                   unfilled_labels=inputs["unfilled"])
