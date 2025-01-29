from .coordinate_system import (
    yaml_to_frame_name, CoordinateSystemModel, yaml_to_coordinate_system,
    cfg_to_coordinate_system)

from dataclasses import dataclass
from typing import Union

import astropy.units as u
from astropy.coordinates import SkyCoord


SKY_CENTER_KEY = 'sky_center'

SHAPE_KEY = 'sdim'

FOV_KEY = 'fov'
FOV_UNIT_KEY = 'fov_unit'
FOV_UNIT_DEFAULT = 'arcsec'

ROTATION_KEY = 'rotation'
ROTATION_DEFAULT = 0.
ROTATION_UNIT_KEY = 'rotation_unit'
ROTATION_UNIT_DEFAULT = 'deg'


@dataclass
class WcsModel:
    center: SkyCoord
    shape: tuple[int, int]
    fov: tuple[u.Quantity, u.Quantity]
    rotation: u.Quantity
    coordinate_system: CoordinateSystemModel


def yaml_to_wcs_model(grid_config: dict) -> WcsModel:
    '''
    Builds the reconstruction grid from the given configuration.

    The reconstruction grid is defined by the world location, field of view
    (FOV), shape (resolution), and rotation, all specified in the input
    configuration. These parameters are extracted from the grid_config dictionary
    using helper functions.

    Parameters
    ----------
    grid_config : dict
        The configuration dictionary containing the following keys:
        - `sky_center`: World coordinate of the spatial grid center.
        - `fov`: Field of view of the grid in appropriate units.
        - `fov_unit`: (Optional) unit for the fov.
        - `sdim`: Shape of the grid, i.e. resolution, as (sdim, sdim).
        - `rotation`: Rotation of the grid.
        - `rotation_unit`: (Optional) unit for the rotation.
        - `energy_bin`: Holding `e_min`, `e_max`, and `reference_bin`.
        - `energy_unit`: The units for `e_min` and `e_max`

    '''
    center = _yaml_to_sky_center(grid_config)
    shape = _yaml_to_shape(grid_config)
    fov = _yaml_to_fov(grid_config)
    rotation = _yaml_to_rotation(grid_config)
    coordinate_system = yaml_to_coordinate_system(grid_config)

    return WcsModel(
        center=center,
        shape=shape,
        fov=fov,
        rotation=rotation,
        coordinate_system=coordinate_system
    )


def cfg_to_wcs_model(grid_config: dict) -> WcsModel:
    '''
    Builds the reconstruction grid from the given configuration.

    The reconstruction grid is defined by the world location, field of view
    (FOV), shape (resolution), and rotation, all specified in the input
    configuration. These parameters are extracted from the grid_config dictionary
    using helper functions.
    '''

    center = cfg_to_sky_center(grid_config)
    shape = _cfg_to_shape(grid_config)
    fov = _cfg_to_fov(grid_config)
    rotation = _cfg_to_rotation(grid_config)
    coordinate_system = cfg_to_coordinate_system(grid_config)

    return WcsModel(
        center=center,
        shape=shape,
        fov=fov,
        rotation=rotation,
        coordinate_system=coordinate_system
    )


def _yaml_to_sky_center(grid_config: dict) -> SkyCoord:
    CENTER_RA_KEY = 'ra'
    DEC_KEY = 'dec'

    CENTER_UNIT_KEY = 'unit'
    CENTER_RA_UNIT_KEY = 'ra_unit'
    CENTER_DEC_UNIT_KEY = 'dec_unit'
    UNIT_DEFAULT = 'deg'

    ra = grid_config[SKY_CENTER_KEY][CENTER_RA_KEY]
    dec = grid_config[SKY_CENTER_KEY][DEC_KEY]
    ra_unit = dec_unit = getattr(
        u, grid_config[SKY_CENTER_KEY].get(CENTER_UNIT_KEY, UNIT_DEFAULT))

    if grid_config[SKY_CENTER_KEY].get(CENTER_RA_UNIT_KEY):
        ra_unit = getattr(u, grid_config[SKY_CENTER_KEY].get(
            CENTER_RA_UNIT_KEY))
    if grid_config[SKY_CENTER_KEY].get(CENTER_DEC_UNIT_KEY):
        dec_unit = getattr(u, grid_config[SKY_CENTER_KEY].get(
            CENTER_DEC_UNIT_KEY))

    frame = yaml_to_frame_name(grid_config)

    return SkyCoord(ra=ra, dec=dec, unit=(ra_unit, dec_unit), frame=frame)


def _yaml_to_shape(grid_config: dict) -> tuple[int, int]:
    """Get the shape from the grid_config."""
    npix = grid_config[SHAPE_KEY]
    return (npix, npix)


def _yaml_to_fov(grid_config: dict) -> tuple[u.Quantity, u.Quantity]:
    """Get the fov from the grid_config."""

    fov = grid_config[FOV_KEY]
    unit = getattr(u, grid_config.get(FOV_UNIT_KEY, FOV_UNIT_DEFAULT))
    return (fov*unit, ) * 2


def _yaml_to_rotation(grid_config: dict) -> u.Quantity:
    """Get the rotation from the grid_config."""
    rotation = grid_config.get(ROTATION_KEY, ROTATION_DEFAULT)
    unit = getattr(u, grid_config.get(
        ROTATION_UNIT_KEY, ROTATION_UNIT_DEFAULT))
    return rotation*unit


def cfg_to_sky_center(sky_cfg: dict[str, Union[str, float]]) -> SkyCoord:
    CENTER_RA_KEY = 'image center ra'
    CENTER_DEC_KEY = 'image center dec'
    CENTER_FRAME_KEY = 'image center frame'
    CENTER_RA_UNIT_KEY = 'image center ra unit'
    CENTER_DEC_UNIT_KEY = 'image center dec unit'

    center_ra = sky_cfg[CENTER_RA_KEY]
    center_dec = sky_cfg[CENTER_DEC_KEY]
    center_frame = sky_cfg[CENTER_FRAME_KEY]

    center_ra_unit = sky_cfg.get(CENTER_RA_UNIT_KEY, 'hourangle')
    center_dec_unit = sky_cfg.get(CENTER_DEC_UNIT_KEY, 'deg')
    ra_unit = getattr(u, center_ra_unit)
    dec_unit = getattr(u, center_dec_unit)
    unit = (ra_unit, dec_unit)

    return SkyCoord(center_ra, center_dec, unit=unit, frame=center_frame)


def _cfg_to_shape(grid_config: dict) -> tuple[int, int]:
    """Get the shape from the grid_config."""
    NPIX_X_KEY = 'space npix x'
    NPIX_Y_KEY = 'space npix y'
    return int(grid_config[NPIX_X_KEY]), int(grid_config[NPIX_Y_KEY])


def _resolve_str_to_unit(s):
    """Convert string of number and unit to radian.

    Support the following units: muas mas as amin deg rad.

    Parameters
    ----------
    s : str
        "muas": u.microarcsecond,
        "mas": u.milliarcsecond,
        "as": u.arcsecond,
        "amin": u.arcmin,
        "deg": u.deg,
        "rad": u.rad,

    """
    units = {
        "muas": u.microarcsecond,
        "mas": u.milliarcsecond,
        "as": u.arcsecond,
        "amin": u.arcmin,
        "deg": u.deg,
        "rad": u.rad,
    }
    keys = list(units.keys())
    keys.sort(key=len)
    for kk in reversed(keys):
        nn = -len(kk)
        unit = s[nn:]
        if unit == kk:
            return float(s[:nn]), units[kk]
    raise RuntimeError("Unit not understood")


def _cfg_to_fov(grid_config: dict) -> tuple[u.Quantity, u.Quantity]:
    '''Convert grid config values from cfg to fov in astropy quantities,
    following the resolve convention.

    Parameters
    ----------
    grid_config: dict
        - "space fov x" : str (Resolve convention)
        - "space fov y" : str (Resolve convention)

    Notes
    -----
    Resolve convention
        "muas": u.microarcsecond,
        "mas": u.milliarcsecond,
        "as": u.arcsecond,
        "amin": u.arcmin,
        "deg": u.deg,
        "rad": u.rad,
    '''
    fov_x, fov_x_unit = _resolve_str_to_unit(grid_config["space fov x"])
    fov_y, fov_y_unit = _resolve_str_to_unit(grid_config["space fov y"])
    return fov_x*fov_x_unit, fov_y*fov_y_unit


def _cfg_to_rotation(grid_config: dict) -> u.Quantity:
    """Get the rotation from the grid_config."""

    CFG_ROTATION_KEY = 'space rotation'
    CFG_ROTATION_UNIT_KEY = 'space rotation unit'

    rotation = grid_config.get(CFG_ROTATION_KEY, ROTATION_DEFAULT)
    unit = getattr(
        u, grid_config.get(CFG_ROTATION_UNIT_KEY, ROTATION_UNIT_DEFAULT))
    return rotation*unit
