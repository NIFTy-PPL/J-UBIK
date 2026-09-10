import pytest

from jubik.parse.wcs.coordinate_system import (
    CoordinateSystemModel, CoordinateSystems)

FRAME_KEY = "frame"
FRAME_DEFAULT = "icrs"
FRAME_EQUINOX_KEY = "equinox"

IMPLEMENTED_FRAMES = ["icrs", "fk5", "fk4", "galactic"]


def test_defaults_from_yaml_dict():
    for frame_name in IMPLEMENTED_FRAMES:
        cs = CoordinateSystemModel.from_yaml_dict({FRAME_KEY: frame_name})
        csc = getattr(CoordinateSystems, frame_name).value
        assert cs == csc


def test_not_implemented():
    not_implemented_system = {FRAME_KEY: "NotExistingCoordinateSystem"}
    with pytest.raises(ValueError):
        CoordinateSystemModel.from_yaml_dict(not_implemented_system)


def test_different_equinox():
    equinox_value = "J1990.0"
    different_equinoxes = dict(
        fk4={FRAME_KEY: "fk4", FRAME_EQUINOX_KEY: equinox_value},
        fk5={FRAME_KEY: "fk5", FRAME_EQUINOX_KEY: equinox_value},
    )

    for name, frame_dict in different_equinoxes.items():
        cs = CoordinateSystemModel.from_yaml_dict(frame_dict)
        assert cs.equinox == equinox_value
        assert cs.radesys == name.upper()

    failing_system = {FRAME_KEY: "icrs", FRAME_EQUINOX_KEY: equinox_value}
    with pytest.raises(ValueError):
        CoordinateSystemModel.from_yaml_dict(failing_system)


def test_coordinate_system_consistency():
    equinox_value = "J1990.0"
    different_equinoxes = dict(
        icrs={FRAME_KEY: "icrs"},
        fk4={FRAME_KEY: "fk4", FRAME_EQUINOX_KEY: equinox_value},
        fk5={FRAME_KEY: "fk5", FRAME_EQUINOX_KEY: equinox_value},
        galactic={FRAME_KEY: "galactic"},
    )

    for name, frame_dict in different_equinoxes.items():
        created = CoordinateSystemModel.from_yaml_dict(frame_dict)
        default = getattr(CoordinateSystems, name).value
        assert created.ctypes == default.ctypes
        assert created.radesys == default.radesys
        if FRAME_EQUINOX_KEY in frame_dict:
            # a custom equinox comes back on a copy; the shared enum member
            # keeps its default so one config cannot change another
            assert created.equinox == equinox_value
            assert default.equinox != equinox_value
        else:
            assert created == default
