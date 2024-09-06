import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

from ...reconstruction_grid import Grid
from ..wcs_subsampling import subsample_grid_centers_in_index_grid


def test_subsample_centers():
    CENTER = SkyCoord(0*u.rad, 0*u.rad)
    MOCK_SHAPE = (24,)*2
    FOV = [MOCK_SHAPE[ii]*(0.5*u.arcsec) for ii in range(2)]
    data_grid = Grid(CENTER, MOCK_SHAPE, FOV)

    RECO_SHAPE = (18,)*2
    reconstruction_grid = Grid(CENTER, RECO_SHAPE, FOV)
    subsample = 3

    subsample_centers = subsample_grid_centers_in_index_grid(
        data_grid.world_extrema(),
        data_grid.wcs,
        reconstruction_grid.wcs,
        subsample)

    # FIXME: This has to changed to an algorithm
    test_subsample_centers = np.load(
        '/home/jruestig/pro/python/j-ubik/tmp/subsample_centers.npy')

    assert np.allclose(test_subsample_centers, subsample_centers)
