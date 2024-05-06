import numpy as np

from jwst import datamodels

from os.path import join

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u


WORLD_LOCATION = (64.66543063107049, -47.86462563973049)  # (ra, dec) in deg
WL = SkyCoord(
    WORLD_LOCATION[0] * u.deg, WORLD_LOCATION[1] * u.deg, frame='icrs')

path = '/home/jruestig/Data/jwst/jw01355_SPT0418-47/jw01355-o016_t001_nircam_clear-f444w'
subfolder = 'jw01355016001_02105_00001_nrcblong'
filename = 'jw01355016001_02105_00001_nrcblong_cal.fits'


fin = datamodels.open(join(path, subfolder, filename))
wtd = fin.meta.wcs.get_transform('world', 'detector')
dtw = fin.meta.wcs.get_transform('detector', 'world')


header = fin.meta.wcs.to_fits()[0]
wcs = WCS(header)
dpix = wcs.world_to_pixel(WL)
wl = wcs.pixel_to_world(dpix[0], dpix[1])


def test_astro(wl):
    return wcs.pixel_to_world(*wcs.world_to_pixel(wl))


def test_jwst(wl):
    return dtw(*wtd(*wl))


wlpy = test_astro(WL)
for ii in range(100):
    wlpy = test_astro(wlpy)
print('Pixel difference (astropy):',
      np.array(wcs.world_to_pixel(wlpy)) - np.array(wcs.world_to_pixel(WL)))

wl = test_jwst(WORLD_LOCATION)
for ii in range(100):
    wl = test_jwst(wl)

print('Pixel difference (jwst):',
      np.array(wtd(*wl)) - np.array(wtd(*WORLD_LOCATION)))
