from jubik.parse.wcs.wcs_model import (
    _get_position_angle,
    WcsModel,
)

from jubik.parse.wcs.coordinate_system import CoordinateSystems

import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord


def test_get_position_angle():
    # Check building
    yaml_dict = {"position_angle": "12uas"}
    ry = _get_position_angle(yaml_dict)
    assert ry == 12 * u.uas

    # Check rotation loading
    yaml_dict_no_unit = {"position_angle": "12"}
    with pytest.raises(ValueError):
        _get_position_angle(yaml_dict_no_unit)

    yaml_dict_wrong_unit = {"position_angle": 12 * u.m}
    with pytest.raises(ValueError):
        _get_position_angle(yaml_dict_wrong_unit)

    # Check defaults
    ry = _get_position_angle({})
    assert ry == 0.0 * u.deg

    with pytest.raises(ValueError, match="use astronomical `position_angle`"):
        _get_position_angle({"rotation": "12deg"})


def test_wcs_model_defaults():
    # Check defaults
    wmy = WcsModel.from_yaml_dict({})

    assert wmy.center == SkyCoord(0.0 * u.deg, 0.0 * u.deg)
    assert wmy.position_angle == 0.0 * u.deg
    assert wmy.coordinate_system == CoordinateSystems.icrs.value


def test_wcs_model_nonstandard():
    # Check defaults
    wmy = WcsModel.from_yaml_dict(
        {
            "sky_center": dict(ra="64deg", dec="32arcsec"),
            "position_angle": "0.1rad",
            "frame": "fk5",
        }
    )

    assert wmy.center == SkyCoord(
        ra="64deg",
        dec="32arcsec",
        frame="fk5",
        equinox=CoordinateSystems.fk5.value.equinox,
    )
    assert wmy.position_angle == 0.1 * u.rad
    assert wmy.coordinate_system == CoordinateSystems.fk5.value
