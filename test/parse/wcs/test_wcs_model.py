from jubik0.parse.wcs.wcs_model import (
    _get_rotation, WcsModel, YAML_ROTATION_KEY, CONFIGPARSER_ROTATION_KEY)

from jubik0.parse.wcs.coordinate_system import CoordinateSystems

import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord


def test_get_rotation():
    # Check building
    yaml_dict = {'rotation': '12uas'}
    config_parser_dict = {'space rotation': '12uas'}
    ry = _get_rotation(yaml_dict, YAML_ROTATION_KEY)
    rc = _get_rotation(config_parser_dict, CONFIGPARSER_ROTATION_KEY)
    assert ry == rc
    assert ry == 12*u.uas

    # Check rotation loading
    yaml_dict_assertion_error = {'rotation': '12'}
    config_parser_dict_assertion_error = {'space rotation': '12'}
    with pytest.raises(AssertionError):
        _get_rotation(yaml_dict_assertion_error, YAML_ROTATION_KEY)
    with pytest.raises(AssertionError):
        _get_rotation(config_parser_dict_assertion_error,
                      CONFIGPARSER_ROTATION_KEY)

    # Check defaults
    ry = _get_rotation({}, YAML_ROTATION_KEY)
    rc = _get_rotation({}, CONFIGPARSER_ROTATION_KEY)
    assert ry == rc
    assert ry == 0.*u.deg


def test_wcs_model_defaults():
    # Check defaults
    wmy = WcsModel.from_yaml_dict({})
    wmp = WcsModel.from_config_parser({})

    assert wmy == wmp
    assert wmy.center == SkyCoord(0.*u.deg, 0.*u.deg)
    assert wmy.rotation == 0.*u.deg
    assert wmy.coordinate_system == CoordinateSystems.icrs.value


def test_wcs_model_nonstandard():
    # Check defaults
    wmy = WcsModel.from_yaml_dict(
        {'sky_center': dict(ra='64deg', dec='32arcsec'),
         'rotation': '0.1rad',
         'frame': 'fk5'}
    )
    wmp = WcsModel.from_config_parser(
        {'image center ra': '64deg',
         'image center dec': '32arcsec',
         'space rotation': '0.1rad',
         'frame': 'fk5'}

    )

    assert wmy == wmp
