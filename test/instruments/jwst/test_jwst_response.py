import numpy as np
import nifty.re as jft
import pytest
from astropy import units as u

from jubik.instruments.jwst.data.jwst_data import DataMetaInformation
from jubik.instruments.jwst.jwst_response import JwstResponse, build_jwst_response
from jubik.instruments.jwst.parse.jwst_response import SkyMetaInformation
from jubik.instruments.jwst.integration.integration import integration_factory
from jubik.instruments.jwst.masking.build_mask import build_mask
from jubik.instruments.jwst.psf.psf_operator import PsfStatic

N = 8
SKY = jft.ShapeWithDtype((N, N), float)
RNG = np.random.default_rng(42)


def identity_sky():
    return jft.Model(lambda x: x["sky"], domain={"sky": SKY})


def delta_kernel(di=0, dj=0):
    kernel = np.zeros((5, 5))
    kernel[2 + di, 2 + dj] = 1.0
    return kernel


def make_response(psf_kernel=None, integrate=lambda x: x, zero_flux=None, mask=None):
    return JwstResponse(
        sky_model=identity_sky(),
        psf=PsfStatic(SKY, psf_kernel),
        unit_conversion=lambda x: x,
        integrate=integrate,
        zero_flux_model=zero_flux,
        mask=build_mask(mask),
    )


def test_identity_pipeline():
    response = make_response(psf_kernel=delta_kernel())
    field = RNG.normal(size=(N, N))
    assert np.allclose(response({"sky": field}), field, atol=1e-10)


def test_none_psf_is_identity():
    response = make_response(psf_kernel=None)
    field = RNG.normal(size=(N, N))
    assert np.allclose(response({"sky": field}), field)


def test_psf_shift_orientation():
    response = make_response(psf_kernel=delta_kernel(di=1))
    field = RNG.normal(size=(N, N))
    out = response({"sky": field})
    assert np.allclose(out[1:], field[:-1], atol=1e-10)


@pytest.mark.parametrize("reduction_factor", [1, 2])
def test_integration_flux_conserves_sum(reduction_factor):
    integrate = integration_factory(
        unit=u.Unit("Jy"),
        high_resolution_shape=(N, N),
        reduction_factor=reduction_factor,
    )
    response = make_response(psf_kernel=delta_kernel(), integrate=integrate)
    field = RNG.normal(size=(N, N))
    out = response({"sky": field})
    assert out.shape == (N // reduction_factor, N // reduction_factor)
    assert np.isclose(out.sum(), field.sum())


@pytest.mark.parametrize("reduction_factor", [1, 2])
def test_integration_surface_brightness_conserves_mean(reduction_factor):
    integrate = integration_factory(
        unit=u.MJy / u.sr,
        high_resolution_shape=(N, N),
        reduction_factor=reduction_factor,
    )
    response = make_response(psf_kernel=delta_kernel(), integrate=integrate)
    field = RNG.normal(size=(N, N))
    out = response({"sky": field})
    assert out.shape == (N // reduction_factor, N // reduction_factor)
    assert np.isclose(out.mean(), field.mean())


def test_mask_selects_pixels():
    mask = np.zeros((N, N), dtype=bool)
    mask[2:4, 2:4] = True
    response = make_response(psf_kernel=delta_kernel(), mask=mask)
    field = RNG.normal(size=(N, N))
    out = response({"sky": field})
    assert out.shape == (4,)
    assert np.allclose(out, field[mask])


def test_zero_flux_added_after_integration():
    zero_flux = jft.Model(
        lambda x: x["zf"], domain={"zf": jft.ShapeWithDtype((1,), float)}
    )
    response = make_response(psf_kernel=delta_kernel(), zero_flux=zero_flux)
    field = RNG.normal(size=(N, N))
    out = response({"sky": field, "zf": np.array([3.0])})
    assert np.allclose(out, field + 3.0)


def test_linearity():
    response = make_response(psf_kernel=delta_kernel())
    field = RNG.normal(size=(N, N))
    assert np.allclose(response({"sky": 2 * field}), 2 * response({"sky": field}))


def build_constant_sky_response(sky_unit, sky_dvol, data_unit, data_dvol, subsample):
    shape = (2 * subsample, 2 * subsample)
    sky = jft.Model(lambda x: x["sky"], domain={"sky": jft.ShapeWithDtype(shape)})
    return build_jwst_response(
        sky_in_subsampled_data=sky,
        data_meta=DataMetaInformation(
            unit=data_unit, dvol=data_dvol, pixel_scale=None, color=None
        ),
        data_subsample=subsample,
        sky_meta=SkyMetaInformation(grid_extension=None, dvol=sky_dvol, unit=sky_unit),
        psf=None,
        zero_flux_model=None,
        data_mask=None,
    ), {"sky": np.ones(shape)}


@pytest.mark.parametrize("subsample", [1, 2, 3])
def test_full_response_flux_sky_conserves_flux(subsample):
    # 1 Jy per (0.1 arcsec)^2 sky pixel gives 0.09 Jy per (0.03 arcsec)^2 data pixel.
    response, x = build_constant_sky_response(
        u.Jy, (0.1 * u.arcsec) ** 2, u.mJy, (0.03 * u.arcsec) ** 2, subsample
    )
    np.testing.assert_allclose(response(x), 90.0)


@pytest.mark.parametrize("subsample", [1, 2])
def test_full_response_surface_brightness_sky_gives_pixel_flux(subsample):
    # A flat 1 MJy/sr sky integrated over a data pixel gives 1 MJy * pixel solid angle.
    response, x = build_constant_sky_response(
        u.MJy / u.sr, (0.1 * u.arcsec) ** 2, u.Jy, (0.03 * u.arcsec) ** 2, subsample
    )
    np.testing.assert_allclose(response(x), 1e6 * (0.03 * np.pi / 648000) ** 2)
