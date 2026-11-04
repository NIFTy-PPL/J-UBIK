"""Per-exposure calibration and a fixed reference exposure in the target response (#167)."""

from types import SimpleNamespace

import jax
import nifty.re as jft
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from jubik.instruments.jwst import jwst_response
from jubik.instruments.jwst.alignment.filter_alignment import FilterAlignment
from jubik.instruments.jwst.data.jwst_data import DataMetaInformation
from jubik.instruments.jwst.jwst_response import (
    TargetResponseInput,
    build_target_response,
)
from jubik.instruments.jwst.parse.jwst_response import SkyMetaInformation
from jubik.instruments.jwst.parse.zero_flux_model import ZeroFluxPriorConfigs
from jubik.instruments.jwst.psf.psf_operator import PsfStatic
from jubik.instruments.jwst.rotation_and_shift.coordinates_correction import (
    ShiftAndRotationCorrection,
    build_coordinates_corrected_for_field,
)
from jubik.parse.models.distributions import prior_config_factory
from jubik.wcs.wcs_astropy import WcsAstropy

N_OBS, N = 3, 4
SKY = jft.ShapeWithDtype((N_OBS, N, N))
RNG = np.random.default_rng(167)
# Value equals the latent, so the expected response needs no prior transform.
IDENTITY_PRIOR = prior_config_factory(["normal", 0.0, 1.0])


def target_response(monkeypatch, calibration_prior=None, zero_flux=None):
    sky = jft.Model(lambda x: x["sky"], domain={"sky": SKY}, target=SKY)
    monkeypatch.setattr(jwst_response, "build_sky_to_subsampled_data", lambda **_: sky)
    return build_target_response(
        TargetResponseInput(
            filter_name="f",
            grid=SimpleNamespace(spatial=None),
            sky_filter=SimpleNamespace(target={"f": None}),
            target_data=SimpleNamespace(
                data=np.zeros(SKY.shape),
                mask=np.ones(SKY.shape, bool),
                subsample=1,
                subsample_centers=None,
                psf=None,
            ),
            filter_meta=DataMetaInformation(
                unit=u.Jy, dvol=1.0 * u.arcsec**2, pixel_scale=None, color=None
            ),
            sky_meta=SkyMetaInformation(
                grid_extension=None, dvol=1.0 * u.arcsec**2, unit=u.Jy
            ),
            rotation_and_shift_algorithm=None,
            zero_flux_prior_configs=(
                None
                if zero_flux is None
                else ZeroFluxPriorConfigs(default=zero_flux, names={})
            ),
            shift_and_rotation_correction=None,
            psf_model=PsfStatic(SKY, None),
            calibration_prior=calibration_prior,
        )
    )


@pytest.mark.parametrize("zero_flux", [None, IDENTITY_PRIOR])
def test_default_adds_no_parameter_keys(monkeypatch, zero_flux):
    response = target_response(monkeypatch, zero_flux=zero_flux)
    expected = {"sky"} if zero_flux is None else {"sky", "f_target_zero_flux"}
    assert set(response.domain) == expected


def test_calibration_key_has_one_factor_per_exposure(monkeypatch):
    response = target_response(
        monkeypatch, calibration_prior=prior_config_factory(["lognormal", 1.0, 0.1])
    )
    assert set(response.domain) == {"sky", "f_target_calibration"}
    assert response.domain["f_target_calibration"].shape == (N_OBS, 1, 1)


@pytest.mark.parametrize("zero_flux", [None, IDENTITY_PRIOR])
def test_calibration_scales_each_exposure_exactly(monkeypatch, zero_flux):
    response = target_response(monkeypatch, IDENTITY_PRIOR, zero_flux)
    sky = RNG.normal(size=SKY.shape)
    factors = np.array([0.5, 1.0, 2.5]).reshape(N_OBS, 1, 1)
    offsets = np.array([1.0, -2.0, 0.3]).reshape(N_OBS, 1, 1)
    x = {"sky": sky, "f_target_calibration": factors}
    expected = factors * sky  # Jy -> Jy, equal pixel volume: unit conversion is 1
    if zero_flux is not None:  # the zero flux is a data-unit offset, not scaled
        x["f_target_zero_flux"] = offsets
        expected = expected + offsets
    np.testing.assert_allclose(response(x), expected.ravel(), rtol=1e-12)


def test_calibration_jit_and_grad(monkeypatch):
    response = target_response(monkeypatch, IDENTITY_PRIOR)
    x = {
        "sky": RNG.normal(size=SKY.shape),
        "f_target_calibration": np.array([0.5, 1.0, 2.5]).reshape(N_OBS, 1, 1),
    }
    np.testing.assert_allclose(jax.jit(response)(x), response(x), rtol=1e-12)

    grad = jax.grad(lambda c: response(x | {"f_target_calibration": c}).sum())(
        x["f_target_calibration"]
    )
    # d/dc_i sum_j c_i sky_ij = sum_j sky_ij
    np.testing.assert_allclose(grad.ravel(), x["sky"].sum(axis=(1, 2)), rtol=1e-12)


def test_fixed_reference_exposure_shift():
    # The existing correction-prior config holds exposure 0 fixed with sigma 0.
    center = SkyCoord(10 * u.deg, -5 * u.deg)
    grid = WcsAstropy(center, shape=21, fov=21 * 0.5 * u.arcsec)
    raw = dict(
        model="shift",
        shift_unit="arcsec",
        rotation_unit="deg",
        default=dict(
            shift=["normal", 0.0, [[0.0], [1.0], [1.0]]],
            rotation=["normal", 0.0, 1.0],
        ),
    )
    prior = FilterAlignment("f").load_correction_prior(raw, N_OBS)
    correction = ShiftAndRotationCorrection("f", prior, center)

    data_wcs = WcsAstropy(center, shape=5, fov=5 * 0.7 * u.arcsec)
    xx, yy = np.meshgrid(np.arange(5), np.arange(5), indexing="xy")
    centers = data_wcs.pixel_to_world(xx, yy)
    coordinates = build_coordinates_corrected_for_field(
        correction, grid, [centers] * N_OBS
    )
    assert set(coordinates.domain) == {"f_shift"}

    xi = RNG.normal(size=(N_OBS, 2))
    jacobian = jax.jacfwd(lambda s: coordinates({"f_shift": s}))(xi)
    # jacobian[obs, yx, i, j, shift_obs, shift_yx]
    per_exposure = np.abs(jacobian).sum(axis=(1, 2, 3, 5))
    assert np.all(per_exposure[:, 0] == 0.0)  # nothing moves the reference
    assert np.all(per_exposure[1:, 1:].diagonal() > 0.0)  # the others are free
    np.testing.assert_array_equal(
        coordinates({"f_shift": xi})[0], coordinates({"f_shift": 0 * xi})[0]
    )
