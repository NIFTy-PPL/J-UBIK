import numpy as np

from .wcs_base import WcsBase
from astropy.coordinates import SkyCoord

from typing import Tuple
from numpy.typing import ArrayLike


def subsample_grid_corners_in_index_grid(
    world_extrema: Tuple[SkyCoord, SkyCoord, SkyCoord, SkyCoord],
    to_be_subsampled_grid_wcs: WcsBase,
    index_grid_wcs: WcsBase,
    subsample: int
) -> ArrayLike:
    '''This function finds the index positions for the corners of the pixels of
    the data_grid inside the reconstruction_grid.


    Parameters
    ----------
    world_extrema: SkyCoord
        The sky/world positions of the extrema inside which to find the pixel
        corners of the data_grid.

    to_be_subsampled_grid_wcs: WcsBase
        The world coordinate system associated with the grid for which to find
        the subsampled pixel corners.

    index_grid_wcs: WcsBase
        The world coordinate system associated with the grid which will be
        indexed into. This will typically be the reconstruction_grid. The
        subsample centers will be in units/indices of this grid.

    subsample:
        The multiplicity of the subsampling along each axis. How many
        sub-pixels will a single pixel in the to_be_subsampled_grid have along
        each axis.
    '''
    ssg_pixcenter_indices = to_be_subsampled_grid_wcs.index_grid_from_wl_extrema(
        world_extrema)

    ps = np.arange(0.5/subsample, 1, 1/subsample) - 0.5
    ms = np.vstack(np.array(np.meshgrid(ps, ps)).T)
    subsample_centers = ms[:, :, None, None] + ssg_pixcenter_indices

    to_edge_shift = 1/subsample/2

    e00, e01, e10, e11 = [
        subsample_centers -
        np.array((i*to_edge_shift, j*to_edge_shift))[None, :, None, None]
        for i, j in zip([1, 1, -1, -1], [1, -1, 1, -1])
    ]

    e00, e01, e10, e11 = [to_be_subsampled_grid_wcs.wl_from_index(e) for
                          e in [e00, e01, e10, e11]]

    # rotation to make them circular for the sparse builder
    return np.array(
        [index_grid_wcs.index_from_wl(e) for e in [e00, e01, e11, e10]])


def subsample_grid_corners_in_index_grid_non_vstack(
    world_extrema: Tuple[SkyCoord, SkyCoord, SkyCoord, SkyCoord],
    to_be_subsampled_grid_wcs: WcsBase,
    index_grid_wcs: WcsBase,
    subsample: int
) -> ArrayLike:
    '''This function finds the index positions for the corners of the pixels of
    the data_grid inside the reconstruction_grid.


    Parameters
    ----------
    world_extrema: SkyCoord
        The sky/world positions of the extrema inside which to find the pixel
        corners of the data_grid.

    to_be_subsampled_grid_wcs: WcsBase
        The world coordinate system associated with the grid for which to find
        the subsampled pixel corners.

    index_grid_wcs: WcsBase
        The world coordinate system associated with the grid which will be
        indexed into. This will typically be the reconstruction_grid. The
        subsample centers will be in units/indices of this grid.

    subsample:
        The multiplicity of the subsampling along each axis. How many
        sub-pixels will a single pixel in the to_be_subsampled_grid have along
        each axis.
    '''
    tbsg_pixcenter_indices = to_be_subsampled_grid_wcs.index_grid_from_wl_extrema(
        world_extrema)

    ps = np.arange(0.5/subsample, 1, 1/subsample) - 0.5
    ms = np.vstack(np.array(np.meshgrid(ps, ps)).T)

    subsample_centers = np.zeros((
        tbsg_pixcenter_indices.shape[0],
        tbsg_pixcenter_indices.shape[1] * subsample,
        tbsg_pixcenter_indices.shape[2] * subsample,
    ))
    for ii, ps in enumerate(ms):
        xx = ii % subsample
        yy = ii // subsample
        subsample_centers[:, xx::subsample, yy::subsample] = (
            tbsg_pixcenter_indices + ps[:, None, None])

    to_edge_shift = 1/subsample/2

    e00, e01, e10, e11 = [
        subsample_centers -
        np.array((i*to_edge_shift, j*to_edge_shift))[:, None, None]
        for i, j in zip([1, 1, -1, -1], [1, -1, 1, -1])
    ]

    e00, e01, e10, e11 = to_be_subsampled_grid_wcs.wl_from_index(
        [e00, e01, e10, e11])

    # rotation to make them circular for the sparse builder
    return np.array(
        index_grid_wcs.index_from_wl([e00, e01, e11, e10])
    )[:, ::-1, :, :]
