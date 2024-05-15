import numpy as np

from astropy.coordinates import SkyCoord
from astropy import units as u

from ...reconstruction_grid import Grid
from ..wcs_subsampling import subsample_grid_centers_in_index_grid


def colinear_subsampling(
    data_shape,
    data_dist,
    reco_dist,
    subsample
):
    factor = [int(dd/rd) for dd, rd in zip(data_dist, reco_dist)]
    pix_center = np.array(np.meshgrid(
        *[np.arange(ds)*f for ds, f in zip(data_shape, factor)]
    )) + np.array([1/f for f in factor])[..., None, None]
    pix_center = pix_center[::-1, :, :]

    d = pix_center[0, 1, 0] - pix_center[0, 0, 0]
    ps = (np.arange(0.5/subsample, 1, 1/subsample) - 0.5) * d
    ms = np.vstack(np.array(np.meshgrid(ps, ps)).T)
    ms = ms[:, ::-1]
    return ms[:, :, None, None] + pix_center


def test_subsample_same_fov():
    center = SkyCoord(0*u.rad, 0*u.rad)
    data_shape = (64,)*2
    data_fov = (1*u.arcsec,)*2
    reco_shape = (128,)*2
    reco_fov = (1*u.arcsec,)*2

    data_grid = Grid(center, shape=data_shape, fov=data_fov)
    reco_grid = Grid(center, reco_shape, reco_fov)

    # Test subsampling of 2
    grid_subs = subsample_grid_centers_in_index_grid(
        data_grid.world_extrema(), data_grid.wcs, reco_grid.wcs, subsample=2)
    coli_subs = colinear_subsampling(
        data_grid.shape, data_grid.distances, reco_grid.distances, subsample=2)
    assert np.allclose(grid_subs, coli_subs, atol=1e-7)

    # Test subsampling of 3
    grid_subs = subsample_grid_centers_in_index_grid(
        data_grid.world_extrema(), data_grid.wcs, reco_grid.wcs, subsample=3)
    coli_subs = colinear_subsampling(
        data_grid.shape, data_grid.distances, reco_grid.distances, subsample=3)
    assert np.allclose(grid_subs, coli_subs, atol=1e-7)


def test_subsample_bigger_data_fov():

    center = SkyCoord(0*u.rad, 0*u.rad)
    data_shape = (256,)*2
    data_fov = (2*u.arcsec,)*2
    reco_shape = (128,)*2
    reco_fov = (1*u.arcsec,)*2

    data_grid = Grid(center, shape=data_shape, fov=data_fov)
    reco_grid = Grid(center, reco_shape, reco_fov)

    grid_subs = subsample_grid_centers_in_index_grid(
        reco_grid.world_extrema(), data_grid.wcs, reco_grid.wcs, subsample=2)
    coli_subs = colinear_subsampling(
        reco_grid.shape, data_grid.distances, reco_grid.distances, subsample=2)

    assert np.allclose(coli_subs-1, grid_subs, atol=1e-7)


def test_subsample_smaller_data_fov():

    center = SkyCoord(0*u.rad, 0*u.rad)
    data_shape = (64,)*2
    data_fov = (1*u.arcsec,)*2
    reco_shape = (128,)*2
    reco_fov = (2*u.arcsec,)*2

    data_grid = Grid(center, shape=data_shape, fov=data_fov)
    reco_grid = Grid(center, reco_shape, reco_fov)

    grid_subs = subsample_grid_centers_in_index_grid(
        reco_grid.world_extrema(), data_grid.wcs, reco_grid.wcs, subsample=2)
    coli_subs = colinear_subsampling(
        reco_grid.shape, data_grid.distances, reco_grid.distances, subsample=2)

    assert np.allclose(coli_subs-1, grid_subs, atol=1e-7)
