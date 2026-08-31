from jubik.parse.wcs.angle import parse_angle

import pytest
from astropy import units as u


DEFAULT = 0.0 * u.deg


def test_default():
    assert parse_angle({}, "rotation", DEFAULT) == DEFAULT


def test_string_with_unit():
    assert parse_angle({"rotation": "12uas"}, "rotation", DEFAULT) == 12 * u.uas
    assert parse_angle({"rotation": "0.1rad"}, "rotation", DEFAULT) == 0.1*u.rad


def test_quantity():
    assert parse_angle({"ra": 3 * u.hourangle}, "ra", DEFAULT) == 3*u.hourangle


def test_sexagesimal():
    ra = parse_angle({"ra": "12h30m10s"}, "ra", DEFAULT)
    assert ra == (12 + 30/60 + 10/3600) * u.hourangle


def test_missing_unit():
    with pytest.raises(ValueError, match="`rotation`"):
        parse_angle({"rotation": "12"}, "rotation", DEFAULT)

    with pytest.raises(ValueError, match="`rotation`"):
        parse_angle({"rotation": 12}, "rotation", DEFAULT)


def test_non_angular_unit():
    with pytest.raises(ValueError, match="`rotation`"):
        parse_angle({"rotation": 12 * u.m}, "rotation", DEFAULT)


def test_unparsable():
    with pytest.raises(ValueError, match="`rotation`"):
        parse_angle({"rotation": "abc"}, "rotation", DEFAULT)
