from .coordinate_system import CoordinateSystemModel
from .sky_center import SkyCenter

import astropy.units as u
from astropy.coordinates import SkyCoord

from dataclasses import dataclass


SKY_CENTER_KEY = 'sky_center'
ROTATION_DEFAULT = 0.*u.deg
YAML_ROTATION_KEY = 'rotation'


def _get_rotation(grid_config: dict) -> u.Quantity:
    """Get the rotation from the grid_config."""

    rotation = u.Quantity(grid_config.get(YAML_ROTATION_KEY, ROTATION_DEFAULT))
    assert rotation.unit != u.dimensionless_unscaled, (
        f'`{YAML_ROTATION_KEY}` should carry a unit.'
    )

    return rotation


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
            Rotation of the wcs. (default `0.0deg`)
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
