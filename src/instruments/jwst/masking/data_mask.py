import numpy as np


def get_mask_from_index_centers_within_rgrid(
    dpixcenter_in_rgrid: np.ndarray, rgrid_shape: tuple[int, int]
) -> np.ndarray:
    """Returns a mask that is true when the dpixcenter_in_rgrid is within the bounds of
    rgrid.

    Parameters
    ----------
    dpixcenter_in_rgrid: np.ndarray
        The data-pixel-centers in units of rgrid pixels.
    rgrid_shape: tuple[int, int]
        The shape of rgrid.
    """
    return (
        (dpixcenter_in_rgrid[0] > 0)
        * (dpixcenter_in_rgrid[1] > 0)
        * (dpixcenter_in_rgrid[0] < rgrid_shape[0])
        * (dpixcenter_in_rgrid[1] < rgrid_shape[1])
    )
