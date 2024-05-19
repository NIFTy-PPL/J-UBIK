from astropy.coordinates import SkyCoord
from astropy import units
from .reconstruction_grid import Grid

from typing import Tuple


def define_world_location(config: dict) -> SkyCoord:
    ra = config['grid']['pointing']['ra']
    dec = config['grid']['pointing']['dec']
    frame = config['grid']['pointing'].get('frame', 'icrs')
    unit = getattr(units, config['grid']['pointing'].get('unit', 'deg'))
    return SkyCoord(ra=ra*unit, dec=dec*unit, frame=frame)


def get_shape(config: dict) -> Tuple[int, int]:
    npix = config['grid']['sdim']
    return (npix, npix)


def get_fov(config: dict) -> Tuple[units.Quantity, units.Quantity]:
    fov = config['grid']['fov']
    unit = getattr(units, config['grid'].get('fov_unit', 'arcsec'))
    return fov*unit


def get_rotation(config: dict) -> units.Quantity:
    rotation = config['grid']['pointing']['rotation']
    unit = getattr(units, config['grid']['pointing'].get('unit', 'deg'))
    return rotation*unit


def build_reconstruction_grid_from_config(config: dict) -> Grid:
    wl = define_world_location(config)
    fov = get_fov(config)
    shape = get_shape(config)
    rotation = get_rotation(config)
    return Grid(wl, shape, (fov.to(units.deg), fov.to(units.deg)), rotation=rotation)


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

    rot_string = 'r' + '_'.join([f'({r}*{rr})' for r, rr in zip(
        config['mock_setup']['rotations'],
        config['mock_setup']['reported_rotations'],
    )])
    shi_string = 's' + '_'.join([f'({hypot(*s)}*{hypot(*rs)})' for s, rs in zip(
        config['mock_setup']['shifts'],
        config['mock_setup']['reported_shifts'],
    )])

    model = config['telescope']['rotation_and_shift']['model']
    subsample = config['telescope']['rotation_and_shift']['subsample']
    method_string = model + f'{subsample}'

    return join(
        config['output'],
        f'{reco_shape}pix',
        f'rot{rot_string}_shf{shi_string}',
        f'{method_string}')
