from .angle import parse_angle
from .coordinate_system import CoordinateSystemModel
from .sky_center import SkyCenter

import astropy.units as u
from astropy.coordinates import SkyCoord

from dataclasses import dataclass


SKY_CENTER_KEY = 'sky_center'
ROTATION_DEFAULT = 0.*u.deg
YAML_ROTATION_KEY = 'rotation'


def _get_rotation(grid_config: dict) -> u.Quantity:
    """Get the rotation of the reconstruction grid against the sky.

    The rotation is the angle between the pixel axes of the reconstruction
    grid and the axes of the celestial coordinate system. It ends up as the
    `PC` matrix, `[[cos, -sin], [sin, cos]]`, of the resulting WCS, see
    `WcsAstropy`.

    With the default of `0deg` the grid is aligned with the coordinate system,
    i.e. the columns follow decreasing right ascension (longitude) and the rows
    increasing declination (latitude). A non-zero rotation turns the grid
    against the sky, which is useful to align the grid with, e.g., an
    elongated source or the scan direction of an instrument.

    Parameters
    ----------
    grid_config : dict
        Configuration which may hold `rotation` as an angle carrying an
        angular unit, e.g. `12deg`. (default `0deg`)

    Returns
    -------
    u.Quantity
        The rotation angle of the grid.
    """

    return parse_angle(grid_config, YAML_ROTATION_KEY, ROTATION_DEFAULT)


@dataclass
class WcsModel:
    center: SkyCoord
    rotation: u.Quantity
    coordinate_system: CoordinateSystemModel

    @classmethod
    def from_yaml_dict(cls, grid_config: dict):
        ''' Builds the reconstruction grid from the given configuration.

        The reconstruction grid is defined by the world location, field of view
        (FOV), shape (resolution), and rotation, all specified in the input
        configuration. These parameters are extracted from the grid_config
        dictionary using helper functions.

        Parameters
        ----------
        sky_center: dict
            World coordinate of the reference pixel (grid center).
        rotation: str
            Rotation of the grid against the sky, i.e. the angle between the
            pixel axes of the grid and the axes of the coordinate system.
            See also `_get_rotation`. (default `0.0deg`)
        frame: str
            See also `CoordinatesSystemModel`. (default `icrs`)
        '''

        rotation = _get_rotation(grid_config)
        coordinate_system = CoordinateSystemModel.from_yaml_dict(grid_config)

        center = SkyCenter.from_yaml_dict(grid_config.get(SKY_CENTER_KEY, {}))

        return WcsModel(
            center=SkyCoord(ra=center.ra,
                            dec=center.dec,
                            frame=coordinate_system.radesys.lower(),
                            equinox=coordinate_system.equinox),
            rotation=rotation,
            coordinate_system=coordinate_system
        )
