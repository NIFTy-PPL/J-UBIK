import numpy as np

from astropy.coordinates import SkyCoord
from astropy import units as u

from jubik0.jwst.reconstruction_grid import Grid

from jubik0.jwst.wcs.wcs_corners import get_pixel_corners
from jubik0.jwst.wcs.wcs_subsampling import get_subsamples_from_wcs


CENTER = SkyCoord(0*u.rad, 0*u.rad)
MOCK_SHAPE = (24,)*2
FOV = [MOCK_SHAPE[ii]*(0.5*u.arcsec) for ii in range(2)]
data_grid = Grid(CENTER, MOCK_SHAPE, FOV)

# Reco sky
RECO_SHAPE = (18,)*2
reconstruction_grid = Grid(CENTER, RECO_SHAPE, FOV)

subsample = 3

pixel_corners = get_pixel_corners(
    data_grid.world_extrema(),
    data_grid.wcs,
    reconstruction_grid.wcs)

subsample_centers = get_subsamples_from_wcs(
    data_grid.world_extrema(),
    data_grid.wcs,
    reconstruction_grid.wcs,
    subsample)


# np.save('./tmp/pixel_corners.npy', pixel_corners)
# np.save('./tmp/subsample_centers.npy', subsample_centers)


test_pixel_corners = np.load('./tmp/pixel_corners.npy')
test_subsample_centers = np.load('./tmp/subsample_centers.npy')

assert np.allclose(test_pixel_corners, pixel_corners)
assert np.allclose(test_subsample_centers, subsample_centers)
