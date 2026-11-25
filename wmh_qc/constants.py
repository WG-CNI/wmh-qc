"""constants that several qc modules share"""

import numpy as np

MM3_PER_ML = 1000.0
MAD_TO_SD = 1.4826                # median absolute deviation -> sd of a normal
# no measurable sharpness change between the two scans
DEFAULT_BLUR_RATIO = 1.0
# structuring element of every 3d morphology step
CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)
