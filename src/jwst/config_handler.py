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


def config_transform(config: dict):
    for key, val in config.items():
        if isinstance(val, str):
            try:
                config[key] = eval(val)
            except:
                continue
        elif isinstance(val, dict):
            config_transform(val)


def define_mock_output(config: dict):
    from numpy import hypot
    from os.path import join

    reco_shape = config['mock_setup']['reco_shape']

    rot_string = 'r' + \
        '_'.join([f'{r}' for r in config['mock_setup']['rotations']])
    shift_string = 's' + \
        '_'.join([f'{hypot(*r)}' for r in config['mock_setup']['shifts']])

    model = config['telescope']['rotation_and_shift']['model']
    subsample = config['telescope']['rotation_and_shift']['subsample']
    method_string = model if model == 'sparse' else model + f'{subsample}'

    return join(
        config['output'],
        f'{reco_shape}pix',
        f'rot{rot_string}_shf{shift_string}',
        f'{method_string}')
