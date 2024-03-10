from jwst import datamodels
import numpy as np
import yaml

import astropy
# import webbpsf

from sys import exit

import gwcs
from astropy import units
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from typing import Tuple


def get_coordinate_system(
    center: SkyCoord,
    fov: Tuple[float, float],
    Npix: Tuple[int, int]
) -> WCS:
    # Create a WCS object
    w = WCS(naxis=2)

    # Set up ICRS system
    w.wcs.crpix = [Npix[0] / 2, Npix[1] / 2]
    w.wcs.cdelt = [-fov[0] / Npix[0], fov[1] / Npix[1]]
    w.wcs.crval = [center.ra.deg, center.dec.deg]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    return w


def define_location(config: dict) -> SkyCoord:
    ra = config['telescope']['pointing']['ra']
    dec = config['telescope']['pointing']['dec']
    frame = config['telescope']['pointing'].get('frame', 'icrs')
    unit = getattr(units, config['telescope']['pointing'].get('unit', 'deg'))

    return SkyCoord(ra=ra*unit, dec=dec*unit, frame=frame)


def get_pixel(wcs: gwcs.wcs, location: SkyCoord, tol=1e-9) -> tuple:
    return wcs.numerical_inverse(location, with_units=True, tolerance=tol)


config = yaml.load(open('JWST_config.yaml', 'r'), Loader=yaml.SafeLoader)
WORLD_LOCATION = define_location(config)
FOV = config['telescope']['fov'] * \
    getattr(units, config['telescope'].get('fov_unit', 'arcsec'))


# defining the reconstruction grid

w = get_coordinate_system(
    WORLD_LOCATION,
    (FOV.to(units.deg).value, FOV.to(units.deg).value),
    (512, 512)
)


exit()

for fltname, flt in config['files']['filter'].items():
    for ii, filepath in enumerate(flt):
        print(fltname, ii, filepath)
        dm = datamodels.open(filepath)

        data = dm.data
        wcs = dm.meta.wcs

        # print(f'x_dist = {x_dist}, y_dist = {y_dist}')

        # ii0 = int(np.round(d_pix[1])) - FIG_SHAPE//2
        # ii1 = int(np.round(d_pix[1])) + FIG_SHAPE//2
        # jj0 = int(np.round(d_pix[0])) - FIG_SHAPE//2
        # jj1 = int(np.round(d_pix[0])) + FIG_SHAPE//2
