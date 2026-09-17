from .wcs_model import WcsModel

import astropy.units as u

from dataclasses import dataclass


@dataclass
class SpatialModel:
    shape_xy: tuple[int, int]
    fov_xy: tuple[u.Quantity, u.Quantity]
    wcs_model: WcsModel

    @classmethod
    def from_yaml_dict(cls, grid_config: dict):
        ''' Builds the reconstruction grid from the given configuration.

        The reconstruction grid is defined by the world location, field of view
        (FOV), shape (resolution), and position angle, all specified in the input
        configuration. These parameters are extracted from the grid_config
        dictionary using helper functions.

        Parameters
        ----------
        grid_config : dict
            The configuration dictionary containing the following keys:
            - `sky_center`: World coordinate of the spatial grid center.
            - `fov`: Field of view of the grid in appropriate units, in
              public coordinate order `(fov_x, fov_y)` (a scalar is
              broadcast to both axes).
            - `shape`: Shape of the grid in public `(nx, ny)` order (a single
              int is broadcast to `(n, n)`).
            - `position_angle`: Astronomical position angle from North through
              East.
            - `energy_bin`: Holding `e_min`, `e_max`, and `reference_bin`.
            - `energy_unit`: The units for `e_min` and `e_max`

        '''
        shape = yaml_dict_to_shape(grid_config)
        fov = yaml_dict_to_fov(grid_config)

        return SpatialModel(
            shape_xy=shape,
            fov_xy=fov,
            wcs_model=WcsModel.from_yaml_dict(grid_config)
        )


def yaml_dict_to_shape(grid_config: dict) -> tuple[int, int]:
    """Get public spatial ``(nx, ny)`` from the ``shape`` config key."""

    if "sdim" in grid_config:
        raise ValueError("`sdim` was removed; use `shape` in public (x, y) order")
    SHAPE_KEY = 'shape'

    npix = grid_config[SHAPE_KEY]
    shape = (npix, npix) if isinstance(npix, int) else tuple(npix)
    if len(shape) != 2:
        raise ValueError(f"shape must contain (nx, ny); got {npix!r}")
    if any(not isinstance(value, int) or value <= 0 for value in shape):
        raise ValueError(f"shape must contain two positive integers; got {npix!r}")
    return shape


def yaml_dict_to_square_size(grid_config: dict, *, consumer: str) -> int:
    """Return the pixel count for a square-only legacy instrument adapter."""

    nx, ny = yaml_dict_to_shape(grid_config)
    if nx != ny:
        raise ValueError(
            f"{consumer} currently requires a square grid; got shape={(nx, ny)}"
        )
    return nx


def yaml_dict_to_fov(grid_config: dict) -> tuple[u.Quantity, u.Quantity]:
    """Get the field of view `fov` from the grid_config."""
    FOV_KEY = 'fov'

    raw_fov = grid_config[FOV_KEY]
    if isinstance(raw_fov, (list, tuple)):
        fov = u.Quantity([u.Quantity(value) for value in raw_fov])
    else:
        fov = u.Quantity(raw_fov)
    if fov.isscalar:
        fov = u.Quantity((fov, fov))
    if fov.shape != (2,):
        raise ValueError(f"fov must contain (fov_x, fov_y); got {fov!r}")
    if not fov.unit.is_equivalent(u.rad):
        raise u.UnitConversionError(f"`{FOV_KEY}` must carry angular units")
    if any(value <= 0 * fov.unit for value in fov):
        raise ValueError(f"fov must contain two positive angular sizes; got {fov!r}")
    return tuple(fov)


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
