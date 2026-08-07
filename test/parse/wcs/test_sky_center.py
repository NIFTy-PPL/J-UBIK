from jubik.parse.wcs.sky_center import SkyCenter

import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from dataclasses import asdict


def test_default():
    scy = SkyCenter.from_yaml_dict({})

    assert scy.ra == 0.0 * u.hourangle
    assert scy.dec == 0.0 * u.deg

    sky_coord = SkyCoord(**asdict(scy))
    assert sky_coord.ra == scy.ra
    assert sky_coord.dec == scy.dec
    assert sky_coord.frame.name == "icrs"


def test_non_default():
    config_yaml = {"ra": "12.0deg", "dec": "12uas"}

    scy = SkyCenter.from_yaml_dict(config_yaml)

    assert scy.ra == 12.0 * u.deg
    assert scy.dec == 12.0 * u.uas


def test_assert_unit():
    config_ra_fail = dict(ra="13", dec="13deg")
    config_dec_fail = dict(ra="13deg", dec="13")

    with pytest.raises(u.UnitsError):
        SkyCenter.from_yaml_dict(config_ra_fail)
    with pytest.raises(u.UnitsError):
        SkyCenter.from_yaml_dict(config_dec_fail)
