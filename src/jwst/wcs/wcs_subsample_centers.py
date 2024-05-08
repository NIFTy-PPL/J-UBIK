from typing import Tuple
from .wcs_base import WcsBase
from astropy.coordinates import SkyCoord
from numpy.typing import ArrayLike


def subsample_grid_centers_in_index_grid(
    world_extrema: Tuple[SkyCoord, SkyCoord, SkyCoord, SkyCoord],
    to_be_subsampled_grid_wcs: WcsBase,
    index_grid_wcs: WcsBase,
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

    to_be_subsampled_grid_wcs: WcsBase
        The world coordinate system associated with the grid to be subsampled.

    index_grid_wcs: WcsBase
        The world coordinate system associated with the grid which will be
        indexed into. This will typically be the reconstruction_grid. The
        subsample centers will be in units/indices of this grid.

    subsample:
        The multiplicity of the subsampling along each axis. How many
        sub-pixels will a single pixel in the to_be_subsampled_grid have along
        each axis.
    '''
    wl_data_subsample_centers = to_be_subsampled_grid_wcs.wl_subsample_centers(
        world_extrema, subsample)
    px_reco_subsample_centers = index_grid_wcs.index_from_wl(
        wl_data_subsample_centers)
    return px_reco_subsample_centers[:, ::-1, :, :]
