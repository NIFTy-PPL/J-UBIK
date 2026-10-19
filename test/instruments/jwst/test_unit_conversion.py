import numpy as np
import pytest
from astropy import units as u

from jubik.instruments.jwst.integration.integration import integration_factory
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


ARCSEC_IN_RAD = np.pi / (180 * 3600)


@pytest.mark.parametrize("subsample", [1, 2, 3])
@pytest.mark.parametrize("data_unit, expected", [(u.Jy, 1.0), (u.mJy, 1e3)])
def test_flux_to_flux_conserves_flux_with_subsampling(subsample, data_unit, expected):
    # Constant 1 Jy per sky pixel, sky and data pixel of 1 arcsec^2 (issue #149).
    conv = build_unit_conversion(
        u.Jy, 1 * u.arcsec**2, data_unit, 1 * u.arcsec**2 / subsample**2
    )
    n = 4 * subsample
    integrate = integration_factory(data_unit, (n, n), subsample)
    np.testing.assert_allclose(integrate(conv(np.ones((n, n)))), expected)


def test_flux_to_flux_scales_with_pixel_area_ratio():
    # 1 Jy per (0.1 arcsec)^2 sky pixel is 0.09 Jy per (0.03 arcsec)^2 pixel.
    conv = build_unit_conversion(u.Jy, (0.1 * u.arcsec) ** 2, u.Jy, DVOL)
    np.testing.assert_allclose(conv(np.array([1.0])), [0.09])


@pytest.mark.parametrize("subsample", [1, 2])
def test_surface_brightness_to_flux(subsample):
    conv = build_unit_conversion(
        u.MJy / u.sr, (0.1 * u.arcsec) ** 2, u.Jy, DVOL / subsample**2
    )
    expected = 1e6 * (0.03 * ARCSEC_IN_RAD) ** 2 / subsample**2
    np.testing.assert_allclose(conv(np.array([1.0])), [expected])


@pytest.mark.parametrize("subsample", [1, 2])
def test_flux_to_surface_brightness(subsample):
    # Flux is per sky pixel, so only the sky pixel area enters.
    sky_dvol = (0.1 * u.arcsec) ** 2
    conv = build_unit_conversion(u.Jy, sky_dvol, u.MJy / u.sr, DVOL / subsample**2)
    expected = 1e-6 / (0.1 * ARCSEC_IN_RAD) ** 2
    np.testing.assert_allclose(conv(np.array([1.0])), [expected])


def test_unsupported_physical_type_rejected():
    with pytest.raises(AssertionError):
        build_unit_conversion(u.m, DVOL, u.Jy, DVOL)
