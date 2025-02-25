import pytest

from jubik0.parse.wcs.coordinate_system import (
    CoordinateSystemModel, CoordinateSystems)

FRAME_KEY = 'frame'
FRAME_DEFAULT = 'icrs'
FRAME_EQUINOX_KEY = 'equinox'

IMPLEMENTED_FRAMES = ['icrs', 'fk5', 'fk4', 'galactic']


def test_defaults_from_config_parser():
    for frame_name in IMPLEMENTED_FRAMES:
        cs = CoordinateSystemModel.from_config_parser({FRAME_KEY: frame_name})
        csc = getattr(CoordinateSystems, frame_name).value
        assert (cs == csc)


def test_defaults_from_yaml_dict():
    for frame_name in IMPLEMENTED_FRAMES:
        cs = CoordinateSystemModel.from_yaml_dict({FRAME_KEY: frame_name})
        csc = getattr(CoordinateSystems, frame_name).value
        assert (cs == csc)


def test_not_implemented():
    not_implemented_system = {FRAME_KEY: "NotExistingCoordinateSystem"}
    with pytest.raises(ValueError):
        CoordinateSystemModel.from_config_parser(not_implemented_system)
    with pytest.raises(ValueError):
        CoordinateSystemModel.from_yaml_dict(not_implemented_system)


def test_different_equinox():
    equinox_value = 'J1990.0'
    different_equinoxes = dict(
        fk4={FRAME_KEY: 'fk4', FRAME_EQUINOX_KEY: equinox_value},
        fk5={FRAME_KEY: 'fk5', FRAME_EQUINOX_KEY: equinox_value}
    )

    for name, frame_dict in different_equinoxes.items():
        cs = CoordinateSystemModel.from_config_parser(frame_dict)
        assert cs.equinox == equinox_value
        assert cs.radesys == name.upper()

    for name, frame_dict in different_equinoxes.items():
        cs = CoordinateSystemModel.from_yaml_dict(frame_dict)
        assert cs.equinox == equinox_value
        assert cs.radesys == name.upper()

    failing_system = {FRAME_KEY: 'icrs', FRAME_EQUINOX_KEY: equinox_value}
    with pytest.raises(ValueError):
        CoordinateSystemModel.from_yaml_dict(failing_system)
    with pytest.raises(ValueError):
        CoordinateSystemModel.from_config_parser(failing_system)
