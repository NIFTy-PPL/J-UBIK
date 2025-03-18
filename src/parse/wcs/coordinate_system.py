from dataclasses import dataclass
from typing import Optional
from enum import Enum

from configparser import ConfigParser

FRAME_KEY = 'frame'
FRAME_DEFAULT = 'icrs'
FRAME_EQUINOX_KEY = 'equinox'


@dataclass
class CoordinateSystemModel:
    """Configuration for a coordinate system."""
    ctypes: tuple[str, str]
    radesys: str
    equinox: Optional[str] = None

    @classmethod
    def from_yaml_dict(cls, grid_config: dict):
        f'''Parsing coordinate system from yaml dict.

        Parameters
        ----------
        {FRAME_KEY}: str  (default {FRAME_DEFAULT})
        {FRAME_EQUINOX_KEY}: float|None (default None)
        '''
        frame = grid_config.get(FRAME_KEY, FRAME_DEFAULT).lower()
        equinox = grid_config.get(FRAME_EQUINOX_KEY)

        _check_if_implemented(frame)
        coordinate_system = getattr(CoordinateSystems, frame)

        if ((equinox is not None) and (coordinate_system in [
                CoordinateSystems.fk4, CoordinateSystems.fk5])):
            coordinate_system.value.equinox = equinox
        elif (equinox is not None):
            raise ValueError('When setting an equinox, one must set either '
                             '`fk4` or `fk5`.')

        return coordinate_system.value

    @classmethod
    def from_config_parser(cls, grid_config: ConfigParser):
        f'''Parsing coordinate system from ConfigParser.

        Parameters
        ----------
        {FRAME_KEY}: str  (default {FRAME_DEFAULT})
        {FRAME_EQUINOX_KEY}: float|None (default None)
        '''
        return cls.from_yaml_dict(grid_config)


class CoordinateSystems(Enum):
    icrs = CoordinateSystemModel(ctypes=('RA---TAN', 'DEC--TAN'),
                                 radesys='ICRS')

    fk5 = CoordinateSystemModel(ctypes=('RA---TAN', 'DEC--TAN'),
                                radesys='FK5',
                                equinox='J2000.0')

    fk4 = CoordinateSystemModel(ctypes=('RA---TAN', 'DEC--TAN'),
                                radesys='FK4',
                                equinox='B1950.0')

    galactic = CoordinateSystemModel(ctypes=('GLON-TAN', 'GLAT-TAN'),
                                     radesys='GALACTIC')


def _check_if_implemented(coordinate_system: str):
    if coordinate_system not in {cs.name for cs in CoordinateSystems}:
        raise ValueError(f"Unsupported coordinate system: {coordinate_system}."
                         f"Supported systems {[c for c in CoordinateSystems]}")


def yaml_to_frame_name(grid_config: dict) -> str:
    frame = grid_config.get(FRAME_KEY, FRAME_DEFAULT)
    return frame.lower()
