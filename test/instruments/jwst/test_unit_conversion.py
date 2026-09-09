import numpy as np
import pytest
from astropy import units as u

from jubik.instruments.jwst.integration.unit_conversion import build_unit_conversion

DVOL = (0.03 * u.arcsec) ** 2


def test_identical_units_is_identity():
    conv = build_unit_conversion(u.MJy / u.sr, DVOL, u.MJy / u.sr, DVOL)
    x = np.arange(4.0)
    np.testing.assert_array_equal(conv(x), x)


def test_same_physical_type_scales():
    conv = build_unit_conversion(u.MJy / u.sr, DVOL, u.Jy / u.sr, DVOL)
    np.testing.assert_allclose(conv(np.array([1.0, 2.0])), [1e6, 2e6])


def test_same_physical_type_flux_scales():
    conv = build_unit_conversion(u.Jy, DVOL, u.mJy, DVOL)
    np.testing.assert_allclose(conv(np.array([1.0])), [1e3])


def test_different_physical_type_not_implemented():
    with pytest.raises(NotImplementedError):
        build_unit_conversion(u.MJy / u.sr, DVOL, u.Jy, DVOL)
    with pytest.raises(NotImplementedError):
        build_unit_conversion(u.Jy, DVOL, u.MJy / u.sr, DVOL)


def test_unsupported_physical_type_rejected():
    with pytest.raises(AssertionError):
        build_unit_conversion(u.m, DVOL, u.Jy, DVOL)
