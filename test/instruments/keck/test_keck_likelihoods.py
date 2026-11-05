"""build_keck_likelihoods on the synthetic NIRC2 files of the loader tests."""

import copy
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
from jubik.instruments.keck.keck_likelihoods import build_keck_likelihoods
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


def _psf_response(tmp_path, kernel, psf_broadening=None):
    """Exposure a's response to the sky, with `kernel` as its PSF file (#175)."""
    cfg = _config(tmp_path)
    fits.PrimaryHDU(kernel).writeto(tmp_path / "psf.fits", overwrite=True)
    cfg["files"]["exposures"] = cfg["files"]["exposures"][:1]
    if psf_broadening is not None:
        cfg["telescope"]["psf_broadening"] = psf_broadening
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
    x = _params(zero)  # the delta prior keeps its unused latent
    expected = reference(x)
    np.testing.assert_allclose(
        broadened(x), expected, atol=1e-8 * np.abs(expected).max()
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
