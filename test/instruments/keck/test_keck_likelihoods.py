"""build_keck_likelihoods on the synthetic NIRC2 files of the loader tests."""

import copy
import logging
import re
import jax
import nifty.re as jft
import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from scipy.optimize import minimize
from scipy.signal import convolve

from jubik.color import Color
from jubik.grid import Grid
from jubik.instruments.jwst.likelihood.likelihood import (
    GaussianLikelihoodBuilder,
    ShotNoiseGaussianLikelihoodBuilder,
)
from jubik.instruments.jwst.parse.zero_flux_model import match_band_key
from jubik.instruments.jwst.psf.psf_operator import (
    PsfGaussianBroadening,
    PsfKernelMix,
    PsfStatic,
)
from jubik.instruments.keck.keck_likelihoods import build_keck_likelihoods
from jubik.models.distributions import build_parametric_prior_from_prior_config
from jubik.parse.models.distributions import prior_config_factory
from jubik.wcs.wcs_astropy import WcsAstropy
from test_nirc2_loader import CENTER, GRID, HEADER, NX, NY, _write

TELESCOPE = {
    "rotation_and_shift": {
        "linear": {"order": 1, "mode": "constant"},
        "correction_priors": {
            "model": "shift",
            "shift_unit": "arcsec",
            "rotation_unit": "deg",
            "default": {
                "rotation": ["delta", 0.0, 0.1],
                "shift": ["normal", 0.0, 0.01],
            },
        },
    },
    "zero_flux": {"default": ["normal", 0.0, 1.0]},
}
SKY_GRID = Grid(GRID, Color([2.0, 2.2, 2.4] * u.um))
SKY_DOMAIN = {"sky": jft.ShapeWithDtype((2, *GRID.shape_yx), float)}


def _config(tmp_path, subsample=1):
    for axis in "XY":
        fits.PrimaryHDU(np.zeros((NY, NX))).writeto(tmp_path / f"{axis}.fits")
    np.savetxt(tmp_path / "band.txt", np.c_[[2.0, 2.4], [1.0, 1.0]])
    profile = np.exp(-0.5 * (np.arange(5) - 2.0) ** 2)
    fits.PrimaryHDU(np.outer(profile, profile)).writeto(tmp_path / "psf.fits")
    rng = np.random.default_rng(0)
    entries = []
    for name, noise in (
        (
            "a",
            {"sigma": str(_write(tmp_path / "sig.fits", np.full((NY, NX), 4.0)).path)},
        ),
        ("b", {"detector_noise": dict(gain=4.0, read_noise=50.0, n_reads=8, sky=10.0)}),
    ):
        science = _write(
            tmp_path / f"{name}.fits", 100 + rng.normal(size=(NY, NX)), HEADER
        )
        entries.append(
            dict(
                science=str(science.path),
                filter="Kp",
                psf=str(tmp_path / "psf.fits"),
                throughput=str(tmp_path / "band.txt"),
                flux_per_count_rate="1e-6",  # yaml.safe_load reads 1e-6 as a string
                atmosphere="airmass 1.0, no extinction",
                margin=2,
                **noise,
            )
        )
    distortion = {"pre_2015": [str(tmp_path / "X.fits"), str(tmp_path / "Y.fits")]}
    telescope = TELESCOPE | {
        "target": {"subsample": subsample},
        "distortion": distortion,
    }
    return {"files": {"exposures": entries}, "telescope": telescope}


def test_sigma_and_detector_noise_evaluate_jit_and_differentiate(tmp_path):
    products = build_keck_likelihoods(_config(tmp_path), SKY_GRID, SKY_DOMAIN)

    assert list(products.sky_filter.bands) == ["keck_nirc2_Kp_a", "keck_nirc2_Kp_b"]
    first, second = (t.builder for t in products.likelihoods)
    assert type(first) is GaussianLikelihoodBuilder
    assert isinstance(second, ShotNoiseGaussianLikelihoodBuilder)
    bands = products.sky_filter.bands
    assert bands["keck_nirc2_Kp_a"] is bands["keck_nirc2_Kp_b"]
    # PsfStatic keeps only the kernel's transform: it is the empirical kernel's,
    # normalised, not the placeholder's
    psf = products.likelihoods[0].builder.response.psf
    profile = np.exp(-0.5 * (np.arange(5) - 2.0) ** 2)
    kernel = np.outer(profile, profile)[None] / np.outer(profile, profile).sum()
    np.testing.assert_allclose(
        psf.kernel_fft, np.fft.rfft2(kernel, psf._fft_shape), atol=1e-12
    )

    likelihood = products.likelihood
    x = jft.random_like(jax.random.PRNGKey(0), likelihood.domain)
    value = likelihood(x)
    assert np.isfinite(value)
    np.testing.assert_allclose(jax.jit(likelihood)(x), value, rtol=1e-6)
    gradient = jax.grad(likelihood)(x)
    assert all(np.all(np.isfinite(g)) for g in jax.tree.leaves(gradient))


def test_empirical_kernel_with_subsample_two_raises(tmp_path):
    with pytest.raises(ValueError, match="pixel response"):
        build_keck_likelihoods(_config(tmp_path, subsample=2), SKY_GRID, SKY_DOMAIN)


# 10 mas detector pixels; the coarse grid has 6 of them per sky pixel
@pytest.mark.parametrize("shape, fov", [(16, 0.16), (4, 0.24)])
@pytest.mark.parametrize("noise", ["sigma", "detector_noise"])
def test_cropping_keeps_the_likelihood(tmp_path, noise, shape, fov):
    """A detector larger than the sky grid, loaded cut to the grid plus a
    margin of the PSF half-width and whole, gives the same likelihood."""
    ny, nx = 64, 72
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[:ny, :nx]
    maps = dict(
        X=0.3 * np.sin(2 * np.pi * xx / nx), Y=0.2 * np.cos(2 * np.pi * yy / ny)
    )
    for axis, shift in maps.items():  # a distortion that varies across the detector
        fits.PrimaryHDU(shift).writeto(tmp_path / f"{axis}.fits")
    np.savetxt(tmp_path / "band.txt", np.c_[[2.0, 2.4], [1.0, 1.0]])
    profile = np.exp(-0.5 * (np.arange(7) - 3.0) ** 2)
    fits.PrimaryHDU(np.outer(profile, profile)).writeto(tmp_path / "psf.fits")
    header = HEADER | {"CRPIX1": 36.0, "CRPIX2": 32.0}
    science = _write(tmp_path / "sci.fits", 100 + rng.normal(size=(ny, nx)), header)
    bad = _write(tmp_path / "bad.fits", rng.random((ny, nx)) < 0.05)
    if noise == "sigma":
        noise_entry = {
            "sigma": str(_write(tmp_path / "sig.fits", 3 + rng.random((ny, nx))).path)
        }
    else:
        noise_entry = {
            "detector_noise": dict(
                gain=4.0,
                read_noise=50.0,
                n_reads=8,
                sky=10 + rng.random((ny, nx)),
                flat=0.9 + 0.2 * rng.random((ny, nx)),
            )
        }
    grid = Grid(WcsAstropy(CENTER, shape=shape, fov=fov * u.arcsec), SKY_GRID.spectral)
    domain = {"sky": jft.ShapeWithDtype((2, shape, shape), float)}

    def build(margin):
        entry = dict(
            science=str(science.path),
            filter="Kp",
            psf=str(tmp_path / "psf.fits"),
            throughput=str(tmp_path / "band.txt"),
            flux_per_count_rate=1e-6,
            atmosphere="airmass 1.0, no extinction",
            bad_pixels=str(bad.path),
            margin=margin,
            **noise_entry,
        )
        distortion = {"pre_2015": [str(tmp_path / "X.fits"), str(tmp_path / "Y.fits")]}
        # wrap: the sky beyond the grid is not zero, so the margin matters
        wrap = TELESCOPE["rotation_and_shift"] | {
            "linear": {"order": 1, "mode": "wrap"}
        }
        telescope = TELESCOPE | {
            "target": {"subsample": 1},
            "distortion": distortion,
            "rotation_and_shift": wrap,
        }
        cfg = {"files": {"exposures": [entry]}, "telescope": telescope}
        return build_keck_likelihoods(cfg, grid, domain)

    cut, whole, no_margin = build(3), build(10_000), build(0)
    (cut_builder,), (whole_builder,) = (
        [t.builder for t in p.likelihoods] for p in (cut, whole)
    )
    assert whole_builder.data.shape == (1, ny, nx)
    assert cut_builder.data.shape[1] < ny and cut_builder.data.shape[2] < nx
    assert cut_builder.mask.sum() == whole_builder.mask.sum() > 0

    x = jft.random_like(jax.random.PRNGKey(2), whole.likelihood.domain)
    x = x.tree if isinstance(x, jft.Vector) else x
    x["sky"] = 1e3 * jax.numpy.abs(x["sky"])  # a bright sky, so b matters
    np.testing.assert_allclose(cut.likelihood(x), whole.likelihood(x), rtol=1e-12)
    if shape == 16:  # the test can fail: without the margin the PSF wings are cut
        assert not np.isclose(no_margin.likelihood(x), whole.likelihood(x), rtol=1e-12)


def test_common_rotation_of_a_nirc2_frame_is_recovered(tmp_path):
    """A frame whose rotator was 0.5° off its header, about 13 mas at 1.5″, is
    fitted back from noise-free data of a ring of knots (#169)."""
    n = 256  # 2.56″ of 10 mas pixels, rotated about 30° on the grid
    for axis in "XY":
        fits.PrimaryHDU(np.zeros((n, n))).writeto(tmp_path / f"{axis}.fits")
    np.savetxt(tmp_path / "band.txt", np.c_[[2.0, 2.4], [1.0, 1.0]])
    profile = np.exp(-0.5 * (np.arange(5) - 2.0) ** 2)
    fits.PrimaryHDU(np.outer(profile, profile)).writeto(tmp_path / "psf.fits")
    header = HEADER | {"CRPIX1": n / 2 + 0.5, "CRPIX2": n / 2 + 0.5}
    entry = dict(
        sigma=str(_write(tmp_path / "sig.fits", np.ones((n, n))).path),
        filter="Kp",
        psf=str(tmp_path / "psf.fits"),
        throughput=str(tmp_path / "band.txt"),
        flux_per_count_rate=1e-6,
        atmosphere="airmass 1.0, no extinction",
        margin=2,
    )
    rotation_and_shift = {
        "linear": {"order": 1, "mode": "constant"},
        "correction_priors": {
            "model": "rshift",
            "shift_unit": "arcsec",
            "rotation_unit": "deg",
            "default": {"rotation": ["delta", 0.0, 0.1], "shift": ["delta", 0.0, 0.1]},
            # the band key, case-insensitive
            "KECK_NIRC2_KP_A": {
                "rotation": ["normal", 0.0, 1.0],
                "shift": ["normal", 0.0, 0.01],
            },
        },
    }
    telescope = {
        "rotation_and_shift": rotation_and_shift,
        "target": {"subsample": 1},
        "distortion": {"pre_2015": [str(tmp_path / f"{a}.fits") for a in "XY"]},
    }
    grid = WcsAstropy(CENTER, shape=128, fov=3.84 * u.arcsec)  # 30 mas pixels
    domain = {"sky": jft.ShapeWithDtype((2, 128, 128), float)}
    band = "keck_nirc2_Kp_a"

    def response(rotator):
        science = _write(
            tmp_path / f"{rotator}.fits",
            np.zeros((n, n)),
            header | {"ROTPOSN": rotator},
        )
        files = {"exposures": [entry | {"science": str(science.path), "exposure": "a"}]}
        cfg = {"files": files, "telescope": copy.deepcopy(telescope)}
        products = build_keck_likelihoods(cfg, Grid(grid, SKY_GRID.spectral), domain)
        (likelihood,) = products.likelihoods
        assert set(likelihood.builder.response.domain) == {
            band,
            f"{band}_shift",
            f"{band}_rotation",
        }
        return lambda x: likelihood.builder.response(
            x | {band: products.sky_filter(x)[band]}
        )

    # knots of 60 mas width on a 1.5″ ring, at uneven angles
    rows, columns = (np.arange(128) - 63.5) * 0.03, (np.arange(128) - 63.5) * 0.03
    yy, xx = np.meshgrid(rows, columns, indexing="ij")
    sky = sum(
        np.exp(
            -0.5 * ((yy - 1.5 * np.sin(a)) ** 2 + (xx - 1.5 * np.cos(a)) ** 2) / 0.06**2
        )
        for a in np.deg2rad([10, 75, 160, 250, 300])
    )
    sky = np.broadcast_to(1e3 * sky, (2, 128, 128))
    truth = {
        "sky": sky,
        f"{band}_shift": np.zeros((1, 2)),
        f"{band}_rotation": np.zeros((1, 1)),
    }
    # the true frame: the rotator was 0.5° beyond its header value
    data = response(HEADER["ROTPOSN"] + 0.5)(truth)
    stated = response(HEADER["ROTPOSN"])

    def chi2(p):
        # latents: the shift prior is N(0, 0.01″), the rotation prior N(0, 1°)
        x = truth | {f"{band}_shift": p[:2][None], f"{band}_rotation": p[2:][None]}
        return ((stated(x) - data) ** 2).sum()

    value_and_grad = jax.jit(jax.value_and_grad(chi2))
    fit = minimize(
        lambda p: [np.asarray(v, float) for v in value_and_grad(p)],
        np.zeros(3),
        jac=True,
        method="BFGS",
    )
    assert chi2(np.zeros(3)) > 1e3 * fit.fun
    np.testing.assert_allclose(fit.x[2], 0.5, atol=0.005)  # deg
    np.testing.assert_allclose(fit.x[:2] * 0.01, 0.0, atol=2e-4)  # arcsec


@pytest.mark.parametrize("rotation", [["normal", 0.0, 0.05], ["delta", 0.5]])
def test_rotation_prior_under_the_shift_model_raises(tmp_path, rotation):
    cfg = _config(tmp_path)
    priors = copy.deepcopy(TELESCOPE["rotation_and_shift"]["correction_priors"])
    priors["keck_nirc2_Kp_a"] = {
        "rotation": rotation,
        "shift": ["normal", 0.0, 0.01],
    }
    cfg["telescope"]["rotation_and_shift"] = TELESCOPE["rotation_and_shift"] | {
        "correction_priors": priors
    }
    with pytest.raises(ValueError, match="keck_nirc2_Kp_a: a rotation prior needs"):
        build_keck_likelihoods(cfg, SKY_GRID, SKY_DOMAIN)


SCALE = 9.952e-3  # arcsec, the pre-2015 plate scale (Yelda et al. 2010)
RING = np.arange(-30, 31)  # pixel offsets of the hand-made kernels


def _gaussian(sigma, offsets):
    g = np.exp(-0.5 * (offsets[:, None] ** 2 + offsets[None, :] ** 2) / sigma**2)
    return g / g.sum()


def _psf_response(tmp_path, kernel, psf_broadening=None, psf_mix=None):
    """Exposure a's response to the sky, with `kernel` as its PSF file (#175);
    `psf_mix` is (second kernel, weight prior) (#176)."""
    cfg = _config(tmp_path)
    fits.PrimaryHDU(kernel).writeto(tmp_path / "psf.fits", overwrite=True)
    cfg["files"]["exposures"] = cfg["files"]["exposures"][:1]
    if psf_broadening is not None:
        cfg["telescope"]["psf_broadening"] = psf_broadening
    if psf_mix is not None:
        second, weight = psf_mix
        fits.PrimaryHDU(second).writeto(tmp_path / "second.fits")
        cfg["files"]["exposures"][0]["psf_mix"] = {
            "psf": str(tmp_path / "second.fits"),
            "weight": weight,
        }
    products = build_keck_likelihoods(cfg, SKY_GRID, SKY_DOMAIN)
    (likelihood,) = products.likelihoods
    response, band = likelihood.builder.response, "keck_nirc2_Kp_a"
    return response, lambda x: response(x | {band: products.sky_filter(x)[band]})


def _params(response):
    domain = response.domain.copy()
    domain.pop("keck_nirc2_Kp_a")  # the sky filter's output
    x = jft.random_like(jax.random.PRNGKey(3), domain | SKY_DOMAIN)
    x = x.tree if isinstance(x, jft.Vector) else dict(x)
    x["sky"] = 1e3 * np.abs(np.asarray(x["sky"]))
    return x


def test_psf_broadening_equals_the_kernel_convolved_by_hand(tmp_path):
    # σ = 2 px: the band-limited Gaussian of PsfGaussianBroadening and the
    # sampled one here differ by exp(-2π²σ²/4) ~ 3e-9 at the Nyquist frequency
    kernel = _gaussian(2.0, np.arange(-15, 16))
    broadened, evaluate = _psf_response(
        tmp_path, kernel, {"default": ["delta", 2.0 * SCALE]}
    )
    by_hand = convolve(kernel, _gaussian(2.0, RING), mode="full")
    (tmp_path / "hand").mkdir()
    _, reference = _psf_response(tmp_path / "hand", by_hand)
    x = _params(broadened)
    expected = reference(x)
    np.testing.assert_allclose(
        evaluate(x), expected, atol=1e-7 * np.abs(expected).max()
    )


def test_zero_psf_broadening_reproduces_the_fixed_kernel(tmp_path):
    kernel = _gaussian(2.0, np.arange(-15, 16))
    fixed, reference = _psf_response(tmp_path, kernel)
    assert set(fixed.domain) == {
        "keck_nirc2_Kp_a",
        "keck_nirc2_Kp_a_shift",
        "keck_nirc2_Kp_a_target_zero_flux",
    }
    (tmp_path / "zero").mkdir()
    zero, broadened = _psf_response(
        tmp_path / "zero", kernel, {"default": ["delta", 0.0]}
    )
    assert isinstance(zero.psf, PsfStatic)
    assert set(zero.domain) == set(fixed.domain)
    x = _params(zero)
    np.testing.assert_array_equal(broadened(x), reference(x))

    # exp(0) = 1″ is not zero: that kernel is broadened
    (tmp_path / "exp").mkdir()
    exp, exp_broadened = _psf_response(
        tmp_path / "exp", kernel, {"default": ["delta", 0.0, 0.0, "exp"]}
    )
    assert isinstance(exp.psf, PsfStatic)  # PsfStatic keeps only the transform
    assert not np.allclose(exp_broadened(x), reference(x))


def test_fixed_psf_broadening_is_the_learned_one_at_that_width(tmp_path):
    kernel = _gaussian(2.0, np.arange(-15, 16))
    fixed, precomputed = _psf_response(
        tmp_path, kernel, {"default": ["delta", 2.0 * SCALE]}
    )
    (tmp_path / "learned").mkdir()
    learned, model = _psf_response(
        tmp_path / "learned", kernel, {"default": ["normal", 2.0 * SCALE, 1e-3]}
    )
    assert isinstance(fixed.psf, PsfStatic)
    assert isinstance(learned.psf, PsfGaussianBroadening)
    x = _params(learned) | {"keck_nirc2_Kp_a_psf_broadening": 0.0}  # σ = its mean
    # PsfStatic's cached transform and fftconvolve round differently
    np.testing.assert_allclose(precomputed(x), model(x), rtol=1e-12)


def test_mixed_psf_broadening_keeps_static_kernels_where_fixed(tmp_path):
    cfg = _config(tmp_path)
    (tmp_path / "static").mkdir()
    reference = build_keck_likelihoods(
        _config(tmp_path / "static"), SKY_GRID, SKY_DOMAIN
    )
    cfg["telescope"]["psf_broadening"] = {
        "default": ["delta", 0.0],
        "keck_nirc2_Kp_b": ["lognormal", 0.01, 0.005],
    }
    products = build_keck_likelihoods(cfg, SKY_GRID, SKY_DOMAIN)
    (a, b), (ref_a, _) = products.likelihoods, reference.likelihoods
    assert isinstance(a.builder.response.psf, PsfStatic)
    assert isinstance(b.builder.response.psf, PsfGaussianBroadening)
    assert set(b.builder.response.domain) >= {"keck_nirc2_Kp_b_psf_broadening"}

    x = jft.random_like(jax.random.PRNGKey(5), products.likelihood.domain)
    x = x.tree if isinstance(x, jft.Vector) else dict(x)
    band = "keck_nirc2_Kp_a"
    sky = {band: products.sky_filter(x)[band]}
    np.testing.assert_array_equal(
        a.builder.response(x | sky), ref_a.builder.response(x | sky)
    )


def test_psf_broadening_of_a_blurred_exposure_is_recovered(tmp_path):
    kernel = _gaussian(2.0, np.arange(-15, 16))
    blurred = convolve(kernel, _gaussian(2.5, RING), mode="full")  # σ = 2.5 px
    _, truth = _psf_response(tmp_path, blurred)
    (tmp_path / "fit").mkdir()
    # band key, case-insensitive; σ ~ 25 ± 15 mas
    response, model = _psf_response(
        tmp_path / "fit",
        kernel,
        {"default": ["delta", 0.0], "KECK_NIRC2_KP_A": ["lognormal", 0.025, 0.015]},
    )
    key = "keck_nirc2_Kp_a_psf_broadening"
    x = _params(response)
    data = truth(x)

    def chi2(latent):
        return ((model(x | {key: latent[0]}) - data) ** 2).sum()

    value_and_grad = jax.jit(jax.value_and_grad(chi2))
    fit = minimize(
        lambda p: [np.asarray(v, float) for v in value_and_grad(p)],
        np.zeros(1),
        jac=True,
        method="BFGS",
    )
    sigma = response.psf.width({key: fit.x[0]}) * SCALE  # arcsec
    np.testing.assert_allclose(sigma, 2.5 * SCALE, rtol=1e-3)


def test_psf_broadening_band_key_matching_no_band_raises(tmp_path):
    with pytest.raises(ValueError, match="psf_broadening"):
        _psf_response(
            tmp_path,
            _gaussian(2.0, np.arange(-15, 16)),
            {"default": ["delta", 0.0], "keck_nirc2_Kp_x": ["delta", 0.0]},
        )


@pytest.mark.parametrize("as_list", [False, True])
def test_detector_noise_flat_from_a_fits_file(tmp_path, as_list):
    cfg = _config(tmp_path)
    flat = np.random.default_rng(177).uniform(0.74, 1.07, (NY, NX))
    path = str(_write(tmp_path / "flat.fits", flat).path)
    noise = cfg["files"]["exposures"][1]["detector_noise"]
    noise["flat"] = [path, 0] if as_list else path
    products = build_keck_likelihoods(cfg, SKY_GRID, SKY_DOMAIN)
    # DN summed over coadds, the grid covers the frame: b = 1 / (gain flat)
    b = products.likelihoods[1].builder.noise.b
    np.testing.assert_allclose(b, 1 / (4.0 * flat))

    noise["flat"] = str(_write(tmp_path / "small.fits", flat[:-1]).path)
    with pytest.raises(ValueError, match="a scalar or the science"):
        build_keck_likelihoods(cfg, SKY_GRID, SKY_DOMAIN)


@pytest.mark.parametrize("weight", [["uniform", -0.5, 2.0], ["delta", 1.5]])
def test_psf_mix_equals_the_kernels_mixed_by_hand(tmp_path, weight):
    sharp, broad = _gaussian(2.0, np.arange(-15, 16)), _gaussian(
        3.0, np.arange(-15, 16)
    )
    mixed, evaluate = _psf_response(tmp_path, sharp, psf_mix=(broad, weight))
    key = "keck_nirc2_Kp_a_psf_mix"
    x = _params(mixed) | {key: 1.0}
    t = _prior_value(key, weight, x)
    assert 1 < t < 2  # past the broad kernel, still positive everywhere
    # a fixed weight takes PsfStatic's cached convolution
    assert type(mixed.psf) is (PsfStatic if weight[0] == "delta" else PsfKernelMix)
    (tmp_path / "hand").mkdir()
    _, reference = _psf_response(tmp_path / "hand", (1 - t) * sharp + t * broad)
    expected = reference(x)
    np.testing.assert_allclose(
        evaluate(x), expected, atol=1e-10 * np.abs(expected).max()
    )


def test_psf_mix_logs_the_negative_flux_at_the_prior_bounds(tmp_path, caplog):
    sharp, broad = _gaussian(2.0, np.arange(-15, 16)), _gaussian(
        3.0, np.arange(-15, 16)
    )
    with caplog.at_level(logging.INFO):
        _psf_response(tmp_path, sharp, psf_mix=(broad, ["uniform", -0.5, 2.0]))
    for t in (-0.5, 2.0):  # negative in the wings, and in the core
        negative = -np.minimum((1 - t) * sharp + t * broad, 0).sum()
        assert negative > 1e-3
        assert f"t = {t:g} has {negative:.2%} of its flux negative" in caplog.text


@pytest.mark.parametrize("fraction", [0.2, 0.8])
def test_core_halo_mix_recovers_the_core_fraction(tmp_path, fraction):
    # a core fraction of 0.2 is broader, 0.8 sharper than the even mix of the two
    halo, core = _gaussian(4.0, np.arange(-15, 16)), _gaussian(1.0, np.arange(-15, 16))
    _, truth = _psf_response(tmp_path, fraction * core + (1 - fraction) * halo)
    (tmp_path / "fit").mkdir()
    response, model = _psf_response(
        tmp_path / "fit", halo, psf_mix=(core, ["uniform", 0.0, 1.0])
    )
    key = "keck_nirc2_Kp_a_psf_mix"
    x = _params(response)
    data = truth(x)

    def chi2(latent):
        return ((model(x | {key: latent[0]}) - data) ** 2).sum()

    value_and_grad = jax.jit(jax.value_and_grad(chi2))
    fit = minimize(
        lambda p: [np.asarray(v, float) for v in value_and_grad(p)],
        np.zeros(1),
        jac=True,
        method="BFGS",
    )
    np.testing.assert_allclose(
        response.psf.weight({key: fit.x[0]}), fraction, rtol=1e-4
    )


def test_psf_mix_of_another_shape_raises(tmp_path):
    kernel, smaller = _gaussian(2.0, np.arange(-15, 16)), _gaussian(2.0, RING[1:-1])
    with pytest.raises(ValueError, match="differ in shape"):
        _psf_response(tmp_path, kernel, psf_mix=(smaller, ["uniform", 0.0, 1.0]))


@pytest.mark.parametrize(
    "weight",
    [["normal", 0.5, 0.5], ["uniform", 0.0, 1.0, "exp"], ["delta", 0.5, 0.0, "exp"]],
)
def test_psf_mix_needs_a_bounded_prior(tmp_path, weight):
    kernel = _gaussian(2.0, np.arange(-15, 16))
    with pytest.raises(ValueError, match="needs a uniform or delta prior"):
        _psf_response(tmp_path, kernel, psf_mix=(kernel, weight))


def _prior_value(key, prior, x):
    """The value of the prior `prior` at the latent `x`, as the builder takes it."""
    return float(
        build_parametric_prior_from_prior_config(key, prior_config_factory(prior))(x)
    )


def _band_limited_gaussian(width, size):
    """PsfGaussianBroadening's Gaussian on a centred odd `size` x `size` grid."""
    f = np.fft.fftfreq(size)
    transfer = np.exp(-2 * np.pi**2 * width**2 * (f[:, None] ** 2 + f[None, :] ** 2))
    return np.fft.fftshift(np.fft.ifft2(transfer).real)


@pytest.mark.parametrize("weight", [["uniform", 0.0, 1.0], ["delta", 0.3]])
@pytest.mark.parametrize("sigma", [["lognormal", 0.02, 0.01], ["delta", 1.5 * SCALE]])
def test_psf_broadening_broadens_the_psf_mix(tmp_path, sigma, weight):
    sharp, broad = _gaussian(2.0, np.arange(-15, 16)), _gaussian(
        3.0, np.arange(-15, 16)
    )
    composed, evaluate = _psf_response(
        tmp_path,
        sharp,
        psf_broadening={"default": sigma},
        psf_mix=(broad, weight),
    )
    mix_key, width_key = "keck_nirc2_Kp_a_psf_mix", "keck_nirc2_Kp_a_psf_broadening"
    learned_t, learned_w = weight[0] != "delta", sigma[0] != "delta"
    # fixed parts go into the kernels once
    model = (
        PsfGaussianBroadening if learned_w else PsfKernelMix if learned_t else PsfStatic
    )
    assert type(composed.psf) is model
    assert (mix_key in composed.domain) == learned_t
    assert (width_key in composed.domain) == learned_w
    x = _params(composed) | {mix_key: 0.4, width_key: -0.3}
    t = _prior_value(mix_key, weight, x)
    width = float(composed.psf.width(x)) if learned_w else 1.5  # σ in pixels
    mix = (1 - t) * sharp + t * broad
    by_hand = convolve(mix, _band_limited_gaussian(width, 31), mode="full")
    (tmp_path / "hand").mkdir()
    _, reference = _psf_response(tmp_path / "hand", by_hand)
    expected = reference(x)
    np.testing.assert_allclose(
        evaluate(x), expected, atol=1e-10 * np.abs(expected).max()
    )


@pytest.mark.parametrize("zero_width", [True, False])
def test_fixed_weight_and_width_take_the_static_path(tmp_path, zero_width):
    sharp, broad = _gaussian(2.0, np.arange(-15, 16)), _gaussian(
        3.0, np.arange(-15, 16)
    )
    mix_key, width_key = "keck_nirc2_Kp_a_psf_mix", "keck_nirc2_Kp_a_psf_broadening"
    weight, sigma = ["uniform", 0.0, 2.0], ["lognormal", 0.02, 0.01]
    learned, evaluate = _psf_response(
        tmp_path,
        sharp,
        psf_broadening=None if zero_width else {"default": sigma},
        psf_mix=(broad, weight),
    )
    x = _params(learned) | {mix_key: 0.7, width_key: -0.4}
    t = _prior_value(mix_key, weight, x)
    sigma = 0.0 if zero_width else _prior_value(width_key, sigma, x)
    (tmp_path / "fixed").mkdir()
    fixed, fixed_evaluate = _psf_response(
        tmp_path / "fixed",
        sharp,
        psf_broadening={"default": ["delta", sigma]},
        psf_mix=(broad, ["delta", t]),
    )
    assert type(fixed.psf) is PsfStatic
    assert not {mix_key, width_key} & set(fixed.domain)
    expected = evaluate(x)
    np.testing.assert_allclose(
        fixed_evaluate(x), expected, atol=1e-10 * np.abs(expected).max()
    )


def test_a_mixed_config_keeps_the_fixed_exposures_static(tmp_path):
    cfg = _config(tmp_path)
    a, b = cfg["files"]["exposures"]
    a["psf_mix"] = {"psf": a["psf"], "weight": ["delta", 0.3]}
    b["psf_mix"] = {"psf": b["psf"], "weight": ["uniform", 0.0, 1.0]}
    cfg["telescope"]["psf_broadening"] = {
        "default": ["delta", 0.0],
        "keck_nirc2_Kp_b": ["lognormal", 0.02, 0.01],
    }
    products = build_keck_likelihoods(cfg, SKY_GRID, SKY_DOMAIN)
    psfs = [likelihood.builder.response.psf for likelihood in products.likelihoods]
    assert [type(psf) for psf in psfs] == [PsfStatic, PsfGaussianBroadening]
    domain = set(products.likelihood.domain)
    assert {"keck_nirc2_Kp_b_psf_mix", "keck_nirc2_Kp_b_psf_broadening"} <= domain
    assert not {"keck_nirc2_Kp_a_psf_mix", "keck_nirc2_Kp_a_psf_broadening"} & domain


@pytest.mark.parametrize(
    "psf_mix",
    [
        {"psf": "second.fits", "prior": ["uniform", 0.0, 1.0]},
        {"psf": "second.fits", "weight": ["uniform", 0.0, 1.0], "extension": 1},
        {"weight": ["uniform", 0.0, 1.0]},
        ["second.fits", ["uniform", 0.0, 1.0]],
    ],
)
def test_psf_mix_of_another_form_raises(tmp_path, psf_mix):
    cfg = _config(tmp_path)
    cfg["files"]["exposures"][0]["psf_mix"] = psf_mix
    with pytest.raises(ValueError, match="psf_mix takes a kernel file, or a mapping"):
        build_keck_likelihoods(cfg, SKY_GRID, SKY_DOMAIN)


def test_band_key_beats_the_longest_pattern():
    keys = {"default", "keck_*", "keck_nirc2_h_*", "keck_nirc2_h_b"}
    assert match_band_key(keys, "keck_nirc2_h_b") == "keck_nirc2_h_b"
    assert match_band_key(keys, "keck_nirc2_h_c") == "keck_nirc2_h_*"
    assert match_band_key(keys, "keck_nirc2_kp_a") == "keck_*"
    assert match_band_key(keys, "keck_nirc2_hcont_a") == "keck_*"
    assert match_band_key(keys, "hst_nic1_f160w_a") is None


PATTERN_PRIORS = {
    "zero_flux": ["normal", 5.0, 1.0],
    "calibration": ["lognormal", 2.0, 0.1],
    "correction_priors": {
        "rotation": ["delta", 0.0, 0.1],
        "shift": ["normal", 0.05, 0.0],  # an offset fixed by sigma 0
    },
    "psf_broadening": ["lognormal", 0.02, 0.01],
}


def _keyed_config(tmp_path):
    """Exposure a in Kp, b in H, with every band-keyed block of the telescope."""
    cfg = copy.deepcopy(_config(tmp_path))
    cfg["files"]["exposures"][1]["filter"] = "H"
    cfg["telescope"]["calibration"] = {"default": ["lognormal", 1.0, 0.1]}
    cfg["telescope"]["psf_broadening"] = {"default": ["delta", 0.0]}
    # unused while no exposure names a second kernel, but its keys are checked
    cfg["telescope"]["psf_mix"] = {"default": ["uniform", 0.0, 1.0]}
    return cfg


def _block(cfg, section):
    telescope = cfg["telescope"]
    return telescope.get(section) or telescope["rotation_and_shift"][section]


@pytest.mark.parametrize("section", PATTERN_PRIORS)
def test_a_filter_pattern_acts_as_the_key_of_each_band_it_covers(tmp_path, section):
    base, products = _keyed_config(tmp_path), []
    for key in (None, "keck_nirc2_h_*", "keck_nirc2_H_b"):
        cfg = copy.deepcopy(base)
        if key is not None:
            _block(cfg, section)[key] = copy.deepcopy(PATTERN_PRIORS[section])
        products.append(build_keck_likelihoods(cfg, SKY_GRID, SKY_DOMAIN))
    likelihoods = [p.likelihood for p in products]
    assert set(likelihoods[1].domain) == set(likelihoods[2].domain)
    x = jft.random_like(jax.random.PRNGKey(3), likelihoods[2].domain)
    default, pattern, band = (
        float(likelihood({k: x[k] for k in likelihood.domain}))
        for likelihood in likelihoods
    )
    assert pattern == band != default
    if section == "psf_broadening":  # H broadened, Kp keeps the static kernel
        psfs = [lh.builder.response.psf for lh in products[1].likelihoods]
        assert [type(psf) for psf in psfs] == [PsfStatic, PsfGaussianBroadening]


@pytest.mark.parametrize("section", [*PATTERN_PRIORS, "psf_mix"])
def test_a_pattern_of_no_band_is_logged_a_band_key_raises(tmp_path, section, caplog):
    cfg = _keyed_config(tmp_path)
    block = _block(cfg, section)
    block["keck_nirc2_j_*"] = block["default"]
    with caplog.at_level(logging.INFO):
        build_keck_likelihoods(cfg, SKY_GRID, SKY_DOMAIN)
    assert f"{section}: keck_nirc2_j_* matches no exposure" in caplog.text
    # _keyed_config's psf_mix block has no exposure with a second kernel to serve
    assert "psf_mix: no exposure names a second kernel" in caplog.text
    for key in ("keck_nirc2_h*", "keck_*_h_*"):  # a band key: no band holds a `*`
        bad = copy.deepcopy(cfg)
        _block(bad, section)[key] = block["default"]
        with pytest.raises(ValueError, match=rf"\['{re.escape(key)}'\] match no band"):
            build_keck_likelihoods(bad, SKY_GRID, SKY_DOMAIN)


def _second_kernel(tmp_path):
    profile = np.exp(-0.5 * ((np.arange(5) - 2.0) / 0.7) ** 2)
    path = tmp_path / "second.fits"
    fits.PrimaryHDU(np.outer(profile, profile)).writeto(path)
    return str(path)


@pytest.mark.parametrize("h_key", ["keck_nirc2_h_*", "keck_nirc2_H_b"])
def test_psf_mix_weight_from_the_telescope_equals_the_entry_weight(tmp_path, h_key):
    base = _keyed_config(tmp_path)
    second = _second_kernel(tmp_path)
    weights = ["uniform", -0.5, 2.0], ["delta", 0.3]  # Kp by default, H by `h_key`
    from_telescope, from_entry = copy.deepcopy(base), copy.deepcopy(base)
    a, b = from_telescope["files"]["exposures"]
    a["psf_mix"], b["psf_mix"] = second, {"psf": second}
    from_telescope["telescope"]["psf_mix"] = {
        "default": weights[0],
        h_key: weights[1],
    }
    for entry, weight in zip(from_entry["files"]["exposures"], weights):
        entry["psf_mix"] = {"psf": second, "weight": weight}
    del from_entry["telescope"]["psf_mix"]
    products = [
        build_keck_likelihoods(cfg, SKY_GRID, SKY_DOMAIN)
        for cfg in (from_telescope, from_entry)
    ]
    psfs = [lh.builder.response.psf for lh in products[0].likelihoods]
    assert [type(psf) for psf in psfs] == [PsfKernelMix, PsfStatic]
    telescope, entry = (p.likelihood for p in products)
    assert set(telescope.domain) == set(entry.domain)
    x = jft.random_like(jax.random.PRNGKey(4), telescope.domain)
    assert float(telescope(x)) == float(entry(x))


@pytest.mark.parametrize("found", ["both", "neither"])
def test_psf_mix_weight_in_both_or_neither_place_raises(tmp_path, found):
    cfg = _keyed_config(tmp_path)
    second = _second_kernel(tmp_path)
    if found == "both":
        weight = ["uniform", 0.0, 1.0]
        cfg["files"]["exposures"][0]["psf_mix"] = {"psf": second, "weight": weight}
    else:
        cfg["files"]["exposures"][0]["psf_mix"] = second
        del cfg["telescope"]["psf_mix"]
    with pytest.raises(ValueError, match=f"psf_mix, found in {found}"):
        build_keck_likelihoods(cfg, SKY_GRID, SKY_DOMAIN)


def test_psf_mix_key_of_an_exposure_without_a_second_kernel_raises(tmp_path):
    cfg = _keyed_config(tmp_path)
    cfg["files"]["exposures"][1]["psf_mix"] = _second_kernel(tmp_path)
    cfg["telescope"]["psf_mix"]["keck_nirc2_kp_a"] = ["delta", 0.3]  # a has none
    match = r"psf_mix: \['keck_nirc2_kp_a'\] .* the exposures with a second kernel"
    with pytest.raises(ValueError, match=match):
        build_keck_likelihoods(cfg, SKY_GRID, SKY_DOMAIN)
