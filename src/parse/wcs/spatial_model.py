from .wcs_model import WcsModel

import astropy.units as u

from dataclasses import dataclass


@dataclass
class SpatialModel:
    shape: tuple[int, int]
    fov: tuple[u.Quantity, u.Quantity]
    wcs_model: WcsModel

    @classmethod
    def from_yaml_dict(cls, grid_config: dict):
        ''' Builds the reconstruction grid from the given configuration.

        The reconstruction grid is defined by the world location, field of view
        (FOV), shape (resolution), and rotation, all specified in the input
        configuration. These parameters are extracted from the grid_config
        dictionary using helper functions.

        Parameters
        ----------
        grid_config : dict
            The configuration dictionary containing the following keys:
            - `sky_center`: World coordinate of the spatial grid center.
            - `fov`: Field of view of the grid in appropriate units.
            - `sdim`: Shape of the grid, i.e. resolution, as (sdim, sdim).
            - `rotation`: Rotation of the grid.
            - `energy_bin`: Holding `e_min`, `e_max`, and `reference_bin`.
            - `energy_unit`: The units for `e_min` and `e_max`

        '''
        shape = yaml_dict_to_shape(grid_config)
        fov = yaml_dict_to_fov(grid_config)

        return SpatialModel(
            shape=shape,
            fov=fov,
            wcs_model=WcsModel.from_yaml_dict(grid_config)
        )

    @classmethod
    def from_config_parser(cls, grid_config: dict):
        '''Builds the reconstruction grid from the given configuration.

        The reconstruction grid is defined by the world location, field of view
        (FOV), shape (resolution), and rotation, all specified in the input
        configuration. These parameters are extracted from the grid_config dictionary
        using helper functions.
        '''

        shape = _cfg_to_shape(grid_config)
        fov = _config_parser_to_fov(grid_config)

        return SpatialModel(
            shape=shape,
            fov=fov,
            wcs_model=WcsModel.from_config_parser(grid_config)
        )


def yaml_dict_to_shape(grid_config: dict) -> tuple[int, int]:
    """Get the spatial shape `sdim` from the grid_config."""

    SHAPE_KEY = 'sdim'

    npix = grid_config[SHAPE_KEY]
    if isinstance(npix, int):
        return (npix, npix)

    if len(npix) == 2:
        return npix

    raise ValueError(f'Only two spatial dimensions. Provided {npix}.')


def yaml_dict_to_fov(grid_config: dict) -> tuple[u.Quantity, u.Quantity]:
    """Get the field of view `fov` from the grid_config."""
    FOV_KEY = 'fov'

    fov = grid_config[FOV_KEY]
    if not (isinstance(fov, int) or isinstance(fov, float)) and len(fov) == 2:
        fov = map(u.Quanitity, fov)
    else:
        fov = (u.Quantity(fov),)*2

    for f in fov:
        assert f.unit != u.dimensionless_unscaled, (
            f'`{FOV_KEY}` should carry a unit.')

    return fov


def _cfg_to_shape(grid_config: dict) -> tuple[int, int]:
    """Get the shape from the grid_config."""
    NPIX_X_KEY = 'space npix x'
    NPIX_Y_KEY = 'space npix y'
    return int(grid_config[NPIX_X_KEY]), int(grid_config[NPIX_Y_KEY])


def _config_parser_to_fov(grid_config: dict) -> tuple[u.Quantity, u.Quantity]:
    '''Convert grid config values from cfg to fov in astropy quantities,
    following the resolve convention.

    Parameters
    ----------
    grid_config: dict
        - "space fov x" : str (Resolve convention)
        - "space fov y" : str (Resolve convention)

    '''
    FOV_KEY_X = "space fov x"
    FOV_KEY_Y = "space fov y"

    fov = (u.Quantity(grid_config[FOV_KEY_X]),
           u.Quantity(grid_config[FOV_KEY_Y]))

    for f, fov_key in zip(fov, [FOV_KEY_X, FOV_KEY_Y]):
        assert f.unit != u.dimensionless_unscaled, (
            f'`{fov_key}` should carry a unit.')

    return fov


def resolve_str_to_quantity(s) -> u.Quantity:
    """Convert string of number and unit to radian.

    Support the following units: muas mas as amin deg rad.

    Parameters
    ----------
    s : str
        "muas": u.microarcsecond,  # TODO: Change to uas
        "mas": u.milliarcsecond,
        "as": u.arcsecond,  # TODO: Change to arcsec
        "amin": u.arcmin,
        "deg": u.deg,
        "rad": u.rad,

    """
    # TODO: Change as->arcsec, and muas->uas. Then one this function is simply:
    # return u.Quantity(s)

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
            return float(s[:nn])*units[kk]
    raise RuntimeError("Unit not understood")
