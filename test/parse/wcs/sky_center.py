from jubik0.parse.wcs.sky_center import SkyCenter

import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from dataclasses import asdict


def test_default():
    scp = SkyCenter.from_config_parser({})
    scy = SkyCenter.from_yaml_dict({})

    assert scp == scy
    assert scp.ra == 0.*u.hourangle
    assert scp.dec == 0.*u.deg

    sky_coord = SkyCoord(**asdict(scp))
    assert sky_coord.ra == scp.ra
    assert sky_coord.dec == scp.dec
    assert sky_coord.frame.name == 'icrs'


def test_non_default():
    config_yaml = {'ra': '12.0deg', 'dec': '12uas'}
    config_parser = {'image center ra': '12.0deg', 'image center dec': '12uas'}

    scy = SkyCenter.from_yaml_dict(config_yaml)
    scp = SkyCenter.from_config_parser(config_parser)

    assert scy == scp
    assert scy.ra == 12.*u.deg
    assert scy.dec == 12.*u.uas


def test_assert_unit():
    config_ra_fail = dict(ra='13', dec='13deg')
    config_dec_fail = dict(ra='13deg', dec='13')

    with pytest.raises(AssertionError):
        SkyCenter.from_yaml_dict(config_ra_fail)
    with pytest.raises(AssertionError):
        SkyCenter.from_yaml_dict(config_dec_fail)

    config_ra_fail = {'image center ra': '13', 'image center dec': '13deg'}
    config_dec_fail = {'image center ra': '13deg', 'image center dec': '13'}

    with pytest.raises(AssertionError):
        SkyCenter.from_config_parser(config_ra_fail)
    with pytest.raises(AssertionError):
        SkyCenter.from_config_parser(config_dec_fail)


def test_equality_of_yaml_and_configparser():
    config_yaml = {'ra': '12.0deg', 'dec': '12uas'}
    config_parser = {'image center ra': '12.0deg', 'image center dec': '12uas'}

    scy = SkyCenter.from_yaml_dict(config_yaml)
    scp = SkyCenter.from_config_parser(config_parser)

    assert scy == scp
    assert scy.ra == 12.*u.deg
    assert scy.dec == 12.*u.uas
