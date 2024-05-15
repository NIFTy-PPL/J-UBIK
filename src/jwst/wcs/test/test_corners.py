import numpy as np

from astropy.coordinates import SkyCoord
from astropy import units as u
from jubik0.jwst.reconstruction_grid import Grid

from ..wcs_corners import get_pixel_corners


def test_pixel_corners():
    CENTER = SkyCoord(0*u.rad, 0*u.rad)
    MOCK_SHAPE = (24,)*2
    FOV = [MOCK_SHAPE[ii]*(0.5*u.arcsec) for ii in range(2)]
    data_grid = Grid(CENTER, MOCK_SHAPE, FOV)

    RECO_SHAPE = (18,)*2
    reconstruction_grid = Grid(CENTER, RECO_SHAPE, FOV)

    pixel_corners = get_pixel_corners(
        data_grid.world_extrema(),
        data_grid.wcs,
        reconstruction_grid.wcs)

    # FIXME: This has to changed to an algorithm
    test_pixel_corners = np.load(
        '/home/jruestig/pro/python/j-ubik/tmp/pixel_corners.npy')
    assert np.allclose(test_pixel_corners, pixel_corners)
