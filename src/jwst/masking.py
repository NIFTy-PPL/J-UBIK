import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple


def mask_index_centers_and_nan(
    dpixcenter_in_rgrid: ArrayLike,
    data: ArrayLike,
    rgrid_shape: Tuple[int, int]
) -> ArrayLike:
    return ((dpixcenter_in_rgrid[0] > 0) *
            (dpixcenter_in_rgrid[1] > 0) *
            (dpixcenter_in_rgrid[0] < rgrid_shape[0]) *
            (dpixcenter_in_rgrid[1] < rgrid_shape[1]) *
            ~np.isnan(data))
