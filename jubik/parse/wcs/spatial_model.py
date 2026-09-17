from .wcs_model import WcsModel

import astropy.units as u

from dataclasses import dataclass

from ..._spatial_validation import normalize_fov, normalize_shape


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
    return normalize_shape(grid_config["shape"], "shape")


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
    return tuple(normalize_fov(grid_config["fov"], "fov"))
