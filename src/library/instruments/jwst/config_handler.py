from typing import Tuple


from astropy.coordinates import SkyCoord
from astropy import units

from .reconstruction_grid import Grid

from typing import Optional


def _get_world_location(config: dict) -> SkyCoord:
    ra = config['grid']['pointing']['ra']
    dec = config['grid']['pointing']['dec']
    frame = config['grid']['pointing'].get('frame', 'icrs')
    unit = getattr(units, config['grid']['pointing'].get('unit', 'deg'))
    return SkyCoord(ra=ra*unit, dec=dec*unit, frame=frame)


def _get_shape(config: dict) -> Tuple[int, int]:
    npix = config['grid']['sdim']
    return (npix, npix)


def _get_fov(config: dict) -> Tuple[units.Quantity, units.Quantity]:
    fov = config['grid']['fov']
    unit = getattr(units, config['grid'].get('fov_unit', 'arcsec'))
    return fov*unit


def _get_rotation(config: dict) -> units.Quantity:
    rotation = config['grid']['pointing']['rotation']
    unit = getattr(units, config['grid']['pointing'].get('unit', 'deg'))
    return rotation*unit


def build_reconstruction_grid_from_config(config: dict) -> Grid:
    wl = _get_world_location(config)
    fov = _get_fov(config)
    shape = _get_shape(config)
    rotation = _get_rotation(config)
    return Grid(wl, shape, (fov.to(units.deg), fov.to(units.deg)), rotation=rotation)


def build_filter_zero_flux(
    config: dict,
    filter: str,
) -> dict:
    prior_config = config['telescope']['zero_flux']
    flower = filter.lower()

    if (flower in prior_config):
        return dict(prior=prior_config[flower])

    return dict(prior=prior_config['prior'])


def build_coordinates_correction_prior_from_config(
    config: dict,
    filter: Optional[str] = '',
    filter_data_set_id: Optional[int] = 0
) -> dict:
    rs_priors = config['telescope']['rotation_and_shift']['priors']

    flower = filter.lower()
    if (flower in rs_priors) and (filter_data_set_id in rs_priors.get(flower, dict())):
        shift = rs_priors[flower][filter_data_set_id]['shift']
        rotation = rs_priors[flower][filter_data_set_id]['rotation']

    else:
        shift = rs_priors['shift']
        rotation = rs_priors['rotation']

    rotation_unit = getattr(units, rs_priors.get('rotation_unit', 'deg'))
    rotation = (rotation[0],
                rotation[1],
                (rotation[2] * rotation_unit).to(units.rad).value)
    return dict(shift=shift, rotation=rotation)


def config_transform(config: dict):
    for key, val in config.items():
        if isinstance(val, str):
            try:
                config[key] = eval(val)
            except:
                continue
        elif isinstance(val, dict):
            config_transform(val)
