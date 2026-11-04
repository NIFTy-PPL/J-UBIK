"""build_keck_likelihoods on the synthetic NIRC2 files of the loader tests."""

import jax
import nifty.re as jft
import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits

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
    kernel = products.likelihoods[0].builder.response.psf.kernel
    assert kernel.shape == (1, 5, 5)  # the empirical kernel, not the placeholder
    np.testing.assert_allclose(kernel.sum(), 1.0)

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


@pytest.mark.parametrize("noise", ["sigma", "detector_noise"])
def test_cropping_keeps_the_likelihood(tmp_path, noise):
    """A detector larger than the sky grid, loaded cut to the grid plus a
    margin of the PSF half-width and whole, gives the same likelihood."""
    ny, nx = 40, 48
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
    header = HEADER | {"CRPIX1": 24.0, "CRPIX2": 20.0}
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
    grid = Grid(WcsAstropy(CENTER, shape=16, fov=0.16 * u.arcsec), SKY_GRID.spectral)
    domain = {"sky": jft.ShapeWithDtype((2, 16, 16), float)}

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
        telescope = TELESCOPE | {"target": {"subsample": 1}, "distortion": distortion}
        cfg = {"files": {"exposures": [entry]}, "telescope": telescope}
        return build_keck_likelihoods(cfg, grid, domain)

    cut, whole = build(3), build(10_000)
    (cut_builder,), (whole_builder,) = (
        [t.builder for t in p.likelihoods] for p in (cut, whole)
    )
    assert whole_builder.data.shape == (1, ny, nx)
    assert cut_builder.data.shape[1] < ny and cut_builder.data.shape[2] < nx
    assert cut_builder.mask.sum() == whole_builder.mask.sum() > 0

    x = jft.random_like(jax.random.PRNGKey(2), whole.likelihood.domain)
    x = x.tree if isinstance(x, jft.Vector) else x
    x["sky"] = 1e3 * jax.numpy.abs(x["sky"])  # a bright sky, so b matters
    np.testing.assert_allclose(cut.likelihood(x), whole.likelihood(x), rtol=1e-6)
