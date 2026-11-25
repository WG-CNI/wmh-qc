"""mm and mm^3 thresholds as voxel counts"""

import numpy as np

from wmh_qc.change import MIN_CLUSTER_MM3, MIN_LESION_MM3
from wmh_qc.harmonisation import (
    AFFINE_MIN_MM3,
    CONTROL_CORTEX_EROSION_MM,
    CONTROL_VENT_EROSION_MM,
    QC_MIN_MM3,
    SPLIT_MIN_HALF_MM3,
    SPLIT_MIN_MM3,
    TAIL_MIN_CORE_MM3,
    TAIL_MIN_WM_MM3,
    WARP_MIN_MM3,
    WARP_MIN_TISSUE_MM3,
)
from wmh_qc.pair_stats import BLUR_MIN_SHELL_MM3, STABLE_MIN_CANDIDATES_MM3, STABLE_MIN_MM3, STABLE_MIN_STRICT_MM3
from wmh_qc.scanner_range import CROP_PAD_MM
from wmh_qc.spacing import axis_steps, morphology_box, voxel_count
from wmh_qc.tissue import BRIGHT_VENT_MIN_EXCESS_MM3, BRIGHT_VENT_MIN_MM3

HALF_MM = (0.5, 0.5, 0.5)
ANISO_MM = (0.5, 0.5, 2.0)
# voxel counts of the paper run on its 0.5 mm iso grid
PAPER_VOXEL_COUNTS = (
    (STABLE_MIN_CANDIDATES_MM3, 1000), (STABLE_MIN_STRICT_MM3, 500), (STABLE_MIN_MM3, 100),
    (BLUR_MIN_SHELL_MM3, 10), (SPLIT_MIN_MM3, 400), (SPLIT_MIN_HALF_MM3, 100), (AFFINE_MIN_MM3, 5000),
    (QC_MIN_MM3, 100), (TAIL_MIN_CORE_MM3, 50), (TAIL_MIN_WM_MM3, 200), (WARP_MIN_TISSUE_MM3, 500),
    (WARP_MIN_MM3, 5000), (MIN_CLUSTER_MM3, 40), (MIN_LESION_MM3, 100),
    (BRIGHT_VENT_MIN_MM3, 2000), (BRIGHT_VENT_MIN_EXCESS_MM3, 1000),
)


def test_paper_grid_gives_exact_integers() -> None:
    """0.5 mm iso: k * 0.125 mm^3 is exactly k voxels, also for a header zoom one float32 ulp off"""
    for count in (1, 10, 40, 50, 100, 200, 400, 500, 1000, 2000, 5000):
        assert voxel_count(count * 0.125, HALF_MM) == count
        assert voxel_count(count * 0.125, (0.49999997, 0.5, 0.50000006)) == count
    assert axis_steps(8.0, HALF_MM) == (16, 16, 16)


def test_paper_grid_keeps_every_published_voxel_count() -> None:
    """every named threshold gives the paper's voxel number on its 0.5 mm grid"""
    for volume_mm3, paper_voxels in PAPER_VOXEL_COUNTS:
        assert voxel_count(volume_mm3, HALF_MM) == paper_voxels
    assert axis_steps(CROP_PAD_MM, HALF_MM) == (16, 16, 16)
    assert morphology_box(CONTROL_CORTEX_EROSION_MM, HALF_MM).shape == (3, 3, 3)
    assert morphology_box(CONTROL_VENT_EROSION_MM, HALF_MM).shape == (5, 5, 5)


def test_counts_scale_with_the_voxel_volume() -> None:
    """40 voxels at 0.5 mm are 5 at 1 mm and 10 at 0.5 x 0.5 x 2 mm"""
    assert voxel_count(5.0, (1.0, 1.0, 1.0)) == 5
    assert voxel_count(5.0, ANISO_MM) == 10
    assert voxel_count(5.0, (3.0, 3.0, 3.0)) == 1


def test_lengths_are_per_axis_and_at_least_one() -> None:
    """thick-slice axis gets fewer steps, never zero"""
    assert axis_steps(8.0, ANISO_MM) == (16, 16, 4)
    assert axis_steps(0.5, (1.0, 1.0, 4.0)) == (1, 1, 1)


def test_morphology_box_is_per_axis_and_never_flat() -> None:
    np.testing.assert_array_equal(morphology_box(1.0, HALF_MM), np.ones((5, 5, 5), bool))
    assert morphology_box(0.5, HALF_MM).shape == (3, 3, 3)
    assert morphology_box(1.0, ANISO_MM).shape == (5, 5, 3)
    assert morphology_box(0.5, (1.0, 1.0, 1.0)).shape == (3, 3, 3)
