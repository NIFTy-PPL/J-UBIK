from dataclasses import dataclass
from typing import Optional
from enum import Enum

FRAME_KEY = 'frame'
FRAME_DEFAULT = 'icrs'
FRAME_EQUINOX_KEY = 'equinox'


@dataclass
class CoordinateSystemModel:
    """Configuration for a coordinate system."""
    ctypes: tuple[str, str]
    radesys: str
    equinox: Optional[float] = None


class CoordinateSystems(Enum):
    icrs = CoordinateSystemModel(ctypes=('RA---TAN', 'DEC--TAN'),
                                 radesys='ICRS')

    fk5 = CoordinateSystemModel(ctypes=('RA---TAN', 'DEC--TAN'),
                                radesys='FK5',
                                equinox=2000.0)

    fk4 = CoordinateSystemModel(ctypes=('RA---TAN', 'DEC--TAN'),
                                radesys='FK4',
                                equinox=1950.0)

    galactic = CoordinateSystemModel(ctypes=('GLON-TAN', 'GLAT-TAN'),
                                     radesys='GALACTIC')


def _check_if_implemented(coordinate_system: str):
    if coordinate_system not in CoordinateSystems:
        raise ValueError(f"Unsupported coordinate system: {coordinate_system}."
                         f"Supported systems {[c for c in CoordinateSystems]}")


def yaml_to_coordinate_system(grid_config: dict) -> CoordinateSystemModel:
    frame = grid_config.get(FRAME_KEY, FRAME_DEFAULT)
    equinox = grid_config.get(FRAME_KEY)

    coordinate_system = getattr(CoordinateSystems, frame)
    _check_if_implemented(coordinate_system)

    if (
        (equinox is not None) and
        (coordinate_system in [CoordinateSystems.fk4, CoordinateSystems.f55])
    ):
        coordinate_system.equinox = equinox

    return coordinate_system.value


def cfg_to_coordinate_system(grid_config: dict) -> CoordinateSystemModel:
    frame = grid_config.get(FRAME_KEY, FRAME_DEFAULT)
    equinox = grid_config.get(FRAME_KEY)

    coordinate_system = getattr(CoordinateSystems, frame)
    _check_if_implemented(coordinate_system)

    if (
        (equinox is not None) and
        (coordinate_system in [CoordinateSystems.fk4, CoordinateSystems.f55])
    ):
        coordinate_system.equinox = equinox

    return coordinate_system.value


def yaml_to_frame_name(grid_config: dict) -> str:
    frame = grid_config.get(FRAME_KEY, FRAME_DEFAULT)
    return frame.lower()
