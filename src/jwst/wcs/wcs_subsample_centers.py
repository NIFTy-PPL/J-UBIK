import numpy as np

from .wcs_base import WcsBase
from astropy.coordinates import SkyCoord

from typing import Tuple
from numpy.typing import ArrayLike


def subsample_grid_centers_in_index_grid(
    world_extrema: Tuple[SkyCoord, SkyCoord, SkyCoord, SkyCoord],
    to_be_subsampled_grid_wcs: WcsBase,
    index_grid_wcs: WcsBase,
    subsample: int
) -> ArrayLike:
    '''This function finds the index positions for the centers of a subsampled
    grid (the to_be_subsampled_grid, typcially the data_grid) inside another
    grid (the index_grid, typically the reconstruction_grid).

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
    minx, maxx, miny, maxy = to_be_subsampled_grid_wcs.index_from_wl_extrema(
        world_extrema)

    ssg_pixcenter_indices = np.array(np.meshgrid(np.arange(minx, maxx, 1),
                                                 np.arange(miny, maxy, 1)))
    ps = np.arange(0.5/subsample, 1, 1/subsample) - 0.5
    ms = np.vstack(np.array(np.meshgrid(ps, ps)).T)
    subsample_centers = ms[:, :, None, None] + ssg_pixcenter_indices

    wl_subsample_centers = to_be_subsampled_grid_wcs.wl_from_index(
        subsample_centers)

    subsample_center_indices = index_grid_wcs.index_from_wl(
        wl_subsample_centers)
    return subsample_center_indices[:, ::-1, :, :]
