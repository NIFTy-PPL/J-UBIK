from astropy.coordinates import SkyCoord
from astropy import units

from typing import Tuple


def define_location(config: dict) -> SkyCoord:
    ra = config['telescope']['pointing']['ra']
    dec = config['telescope']['pointing']['dec']
    frame = config['telescope']['pointing'].get('frame', 'icrs')
    unit = getattr(units, config['telescope']['pointing'].get('unit', 'deg'))
    return SkyCoord(ra=ra*unit, dec=dec*unit, frame=frame)


def get_shape(config: dict) -> Tuple[int, int]:
    npix = config['grid']['npix']
    return (npix, npix)


def get_fov(config: dict) -> Tuple[units.Quantity, units.Quantity]:
    fov = config['telescope']['fov']
    unit = getattr(units, config['telescope'].get('fov_unit', 'arcsec'))
    return fov*unit
