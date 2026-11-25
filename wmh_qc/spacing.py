"""physical sizes (mm, mm^3, ml) on a voxel grid

every threshold is calibrated on the paper's 0.5 mm iso grid: the counts there are exactly the old voxel numbers
"""

import numpy as np

from wmh_qc.constants import MM3_PER_ML
from wmh_qc.containers import Zooms


def voxel_mm3(zooms_mm: Zooms) -> float:
    return float(np.prod(zooms_mm))


def voxel_ml(zooms_mm: Zooms) -> float:
    return voxel_mm3(zooms_mm) / MM3_PER_ML


def voxel_count(volume_mm3: float, zooms_mm: Zooms) -> int:
    # nearest, not ceil: a header zoom of 0.49999997 must still give the 0.5 mm count
    return max(1, round(volume_mm3 / voxel_mm3(zooms_mm)))


def axis_steps(length_mm: float, zooms_mm: Zooms) -> tuple[int, ...]:
    return tuple(max(1, round(length_mm / float(zoom))) for zoom in zooms_mm)


def morphology_box(length_mm: float, zooms_mm: Zooms) -> np.ndarray:
    """box structuring element reaching length_mm along each axis"""
    # on an iso grid this is the same erosion as n passes of the 3x3x3 cube.
    # per axis, so a thick-slice axis is not eroded by the in-plane voxel count
    return np.ones(tuple(2 * steps + 1 for steps in axis_steps(length_mm, zooms_mm)), dtype=bool)
