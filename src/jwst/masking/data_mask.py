from numpy.typing import ArrayLike
from typing import Tuple


def get_mask_from_index_centers(
    dpixcenter_in_rgrid: ArrayLike,
    rgrid_shape: Tuple[int, int]
) -> ArrayLike:
    return (
        (dpixcenter_in_rgrid[0] > 0) *
        (dpixcenter_in_rgrid[1] > 0) *
        (dpixcenter_in_rgrid[0] < rgrid_shape[0]) *
        (dpixcenter_in_rgrid[1] < rgrid_shape[1])
    )
