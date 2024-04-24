from typing import Tuple
from .wcs_base import WcsBase
from astropy.coordinates import SkyCoord
from numpy.typing import ArrayLike


def get_subsamples_from_wcs(
    world_extrema: Tuple[SkyCoord, SkyCoord, SkyCoord, SkyCoord],
    data_grid_wcs: WcsBase,
    reconstruction_grid_wcs: WcsBase,
    subsample: int
) -> ArrayLike:
    '''This function finds the index positions for the centers of the data_grid
    inside the reconstruction_grid.

    Parameters
    ----------
    world_extrema: SkyCoord
        The sky/world positions of the extrema inside which to find the
        subsampling centers.
        Works also if they are outside the grids.

    data_grid_wcs: WcsBase
        The world coordinate system associated with the grid to be subsampled.

    reconstruction_grid_wcs: WcsBase
        The world coordinate system associated with the grid which will be
        indexed into. Hence, the subsample centers will be given with respect
        to the indices of the reconstruction_grid.

    subsample:
        The multiplicity of the subsampling along each dimension.
        How many sub-pixels will a single pixel in the data_grid be having
        along one dimension.
    '''
    wl_data_subsample_centers = data_grid_wcs.wl_subsample_centers(
        world_extrema, subsample)
    px_reco_subsample_centers = reconstruction_grid_wcs.index_from_wl(
        wl_data_subsample_centers)
    return px_reco_subsample_centers[:, ::-1, :, :]
