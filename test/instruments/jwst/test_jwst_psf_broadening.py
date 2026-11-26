"""``psf_broadening`` in `build_jwst_likelihoods`: one learnable Gaussian PSF width
per exposure, on the fake observation of `test_jwst_likelihoods`."""

import jax
import nifty.re as jft
import numpy as np
import pytest
from scipy.signal import convolve

from jubik.instruments.jwst.data.jwst_information import get_pixel_distance
from jubik.instruments.jwst.jwst_likelihoods import build_jwst_likelihoods
from jubik.instruments.jwst.parse.jwst_psf import JwstPsfKernelConfig
from jubik.instruments.jwst.psf import jwst_kernel
from jubik.instruments.jwst.psf.psf_correction import PsfCorrection
from jubik.instruments.jwst.psf.psf_operator import PsfGaussianBroadening, PsfStatic
from jubik.models.distributions import build_parametric_prior_from_prior_config
from jubik.parse.models.distributions import prior_config_factory

from test_jwst_likelihoods import (
    FILTER,
    SKY_DOMAIN,
    FakeJwstData,
    make_config,
    make_grid,
    patched_seams,  # noqa: F401
)

BAND = FILTER.lower()
KEY = f"{BAND}_psf_broadening"
SUBSAMPLE = 2  # make_config's target.subsample
SUBPIXEL = get_pixel_distance(FILTER).to_value("arcsec") / SUBSAMPLE  # arcsec
LOGNORMAL = ["lognormal", 2.0 * SUBPIXEL, 0.5 * SUBPIXEL]


def wide_build_webb_psf(camera, filter, center_pixel, webbpsf_path, subsample, *a, **k):
    """A Gaussian core with power-law wings on a grid wide enough that a Gaussian of
    σ = 2 pixels hardly wraps on it (`PsfGaussianBroadening`'s shortcut)."""
    r = np.hypot(*np.mgrid[-20:21, -20:21])
    psf = np.exp(-0.5 * (r / subsample) ** 2) + 0.01 / (1 + (r / subsample) ** 3)
    return psf / psf.sum()


@pytest.fixture
def broad_kernel(monkeypatch, patched_seams):  # noqa: F811
    monkeypatch.setattr(jwst_kernel, "build_webb_psf", wide_build_webb_psf)


def _gaussian(sigma, radius=12):
    offsets = np.arange(-radius, radius + 1)
    g = np.exp(-0.5 * (offsets[:, None] ** 2 + offsets[None, :] ** 2) / sigma**2)
    return g / g.sum()


def _build(tmp_path, broadening=None, n_files=1, psf=None):
    """The filter's response and a function of the latents evaluating it."""
    tmp_path.mkdir(exist_ok=True)
    cfg = make_config(tmp_path, n_files=n_files)
    cfg["telescope"]["psf"] |= psf or {}
    if broadening is not None:
        cfg["telescope"]["psf_broadening"] = broadening
    products = build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN)
    (target,) = products.target.likelihoods
    response, sky_filter = target.builder.response, products.target.sky_filter
    return response, lambda x: response(x | sky_filter(x)), products, cfg


def _latents(products, seed=3):
    x = jft.random_like(jax.random.PRNGKey(seed), products.target.likelihood.domain)
    x = x.tree if isinstance(x, jft.Vector) else dict(x)
    x["sky"] = 1e3 * np.abs(np.asarray(x["sky"]))
    return x


def test_no_key_and_a_zero_delta_keep_the_response(broad_kernel, tmp_path):
    fixed, reference, products, _ = _build(tmp_path, n_files=2)
    zero, model, _, _ = _build(tmp_path, {"default": ["delta", 0.0]}, n_files=2)
    assert isinstance(zero.psf, PsfStatic)
    assert KEY not in zero.domain and set(zero.domain) == set(fixed.domain)
    x = _latents(products)
    np.testing.assert_array_equal(model(x), reference(x))


@pytest.mark.parametrize("option", ["plain", "correction", "pixel_box"])
def test_fixed_width_is_the_loaded_kernel_convolved_once(
    broad_kernel, monkeypatch, tmp_path, option
):
    """σ = 2 subsampled pixels: the band-limited Gaussian of `PsfGaussianBroadening`
    and the sampled one here differ by exp(-2π²σ²/4) ~ 3e-9 at the Nyquist
    frequency. The kernel is the loader's, with the correction or the sub-pixel box
    (!312; on a branch without it the key is not read)."""
    psf = {}
    if option == "correction":
        texture = 0.1 * np.random.default_rng(0).normal(size=(5, 5))
        PsfCorrection(BAND, SUBSAMPLE, texture, width=0.5).save(tmp_path / "c.npz")
        psf["correction"] = {BAND: str(tmp_path / "c.npz")}
    elif option == "pixel_box":
        psf["pixel_box"] = True
    sigma = 2.0  # subsampled pixels
    _, broadened, products, cfg = _build(
        tmp_path / "fit", {"default": ["delta", sigma * SUBPIXEL]}, psf=psf
    )
    loaded = jwst_kernel.load_psf_kernel(
        jwst_data=FakeJwstData(cfg["files"]["filter"][BAND][0]),
        subsample=SUBSAMPLE,
        target_center=make_grid().spatial.center,
        config_parameters=JwstPsfKernelConfig.from_yaml_dict(cfg["telescope"]["psf"]),
    )
    by_hand = convolve(loaded, _gaussian(sigma), mode="full")
    monkeypatch.setattr(jwst_kernel, "build_webb_psf", lambda *a, **k: by_hand)
    _, reference, _, _ = _build(tmp_path / "hand")
    x = _latents(products)
    expected = reference(x)
    np.testing.assert_allclose(
        broadened(x), expected, atol=1e-9 * np.abs(expected).max()
    )


def test_each_exposure_has_its_own_width(broad_kernel, tmp_path):
    response, model, products, _ = _build(
        tmp_path / "fit", {"default": LOGNORMAL}, n_files=2
    )
    assert isinstance(response.psf, PsfGaussianBroadening)
    assert response.domain[KEY].shape == (2,)
    x = _latents(products) | {KEY: np.array([-1.0, 1.5])}
    # the prior as sptLensing's nir_imaging summary reads it, per exposure
    width = build_parametric_prior_from_prior_config(
        KEY, prior_config_factory(LOGNORMAL), as_model=True
    )
    sigmas = np.asarray(width(x))  # arcsec
    assert sigmas[0] < 0.7 * sigmas[1]
    # the masked response: the unmasked pixels of exposure 0, then of exposure 1
    split = [int(products.target.likelihoods[0].builder.mask[0].sum())]
    out = np.split(model(x), split)
    for e, sigma in enumerate(sigmas):
        _, fixed, _, _ = _build(
            tmp_path / f"fixed{e}", {"default": ["delta", float(sigma)]}, n_files=2
        )
        # PsfStatic's cached transform and fftconvolve round differently
        expected = np.split(fixed(x), split)[e]
        np.testing.assert_allclose(out[e], expected, rtol=1e-10)


def test_width_gradient_is_finite_and_nonzero(broad_kernel, tmp_path):
    _, _, products, _ = _build(tmp_path, {"default": LOGNORMAL}, n_files=2)
    likelihood = products.target.likelihood
    x = _latents(products)
    grad = jax.grad(lambda w: likelihood(x | {KEY: w}))(np.array([0.3, -0.4]))
    assert np.all(np.isfinite(grad)) and np.all(grad != 0)


def test_filter_key_matching_no_filter_raises(patched_seams, tmp_path):  # noqa: F811
    with pytest.raises(ValueError, match=r"psf_broadening: \['f277w'\] match no"):
        _build(tmp_path, {"default": ["delta", 0.0], "F277W": LOGNORMAL})


def test_filter_key_overrides_the_default(broad_kernel, tmp_path):
    response, _, _, _ = _build(tmp_path, {"default": ["delta", 0.0], FILTER: LOGNORMAL})
    assert isinstance(response.psf, PsfGaussianBroadening)
