from typing import Tuple
from .wcs_base import WcsBase
from astropy.coordinates import SkyCoord
from numpy.typing import ArrayLike


def get_pixel_corners(
    world_extrema: Tuple[SkyCoord, SkyCoord, SkyCoord, SkyCoord],
    data_grid_wcs: WcsBase,
    reconstruction_grid_wcs: WcsBase
) -> ArrayLike:
    '''This function finds the index positions for the corners of the pixels of 
    the data_grid inside the reconstruction_grid.

    Parameters
    ----------
    world_extrema: SkyCoord
        The sky/world positions of the extrema inside which to find the pixel
        corners of the data_grid.

    data_grid_wcs: WcsBase
        The world coordinate system associated with the grid for which to find
        the pixel corners.

    reconstruction_grid_wcs: WcsBase
        The world coordinate system associated with the grid which will be
        indexed into.
    '''
    _, (e00, e01, e10, e11) = data_grid_wcs.wl_pixelcenter_and_edges(world_extrema)
    return reconstruction_grid_wcs.index_from_wl(
        [e00, e01, e11, e10])  # needs to be circular for sparse builder
