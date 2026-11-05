"""build_hst_likelihoods and the shared per-exposure builder on the synthetic
NICMOS and WFPC2 files of the loader tests."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import matplotlib.image
import nifty.re as jft
import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS

from jubik.grid import Grid
from jubik.instruments.hst.hst_data import load_hst_exposure
from jubik.instruments.hst.hst_likelihoods import build_hst_likelihoods
from jubik.instruments.hst.tinytim import HstPsfKernel
from jubik.instruments.jwst.exposure_likelihoods import (
    Exposure,
    build_exposure_likelihoods,
)
from jubik.instruments.jwst.likelihood.likelihood import (
    ShotNoise,
    ShotNoiseGaussianLikelihoodBuilder,
)
from jubik.instruments.jwst.plotting import residuals as residual_plots
from jubik.instruments.jwst.plotting.residuals import SkyResiduals
from jubik.instruments.jwst.psf.psf_operator import PsfGaussianBroadening
from jubik.wcs import WcsAstropy
from test_hst_data import CURVE, NIC2_SCALE, SHAPE, SPECTRAL, _grid, _nicmos, _wfpc2

TELESCOPE = {
    "target": {"subsample": 1},
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
    "calibration": {"default": ["lognormal", 1.0, 0.1]},
}
NIC2 = "hst_NIC2_F160W_"


def _kernel_file(path, subsample=1, size=5, scale=0.076):
    kernel = np.exp(
        -0.5 * (np.arange(size * subsample) - (size * subsample - 1) / 2) ** 2
    )
    hdu = fits.PrimaryHDU(np.outer(kernel, kernel))
    hdu.header.update(PIXSCALE=scale / subsample, SUBSAMPL=subsample, CHGDIFF=False)
    hdu.writeto(path)
    return str(path)


def _config(tmp_path, entries):
    np.savetxt(tmp_path / "curve.txt", np.c_[CURVE.wavelength.value, CURVE.throughput])
    psf = _kernel_file(tmp_path / "psf.fits")
    common = dict(throughput=str(tmp_path / "curve.txt"), psf=psf, dq_bits=0)
    return {
        "files": {"exposures": [common | e for e in entries]},
        "telescope": TELESCOPE,
    }


def _nicmos_entry(tmp_path, name, margin=2):
    path = _nicmos(tmp_path / f"{name}_cal.fits")
    return dict(path=str(path), camera="NIC2", filter="F160W", margin=margin)


def _sky_domain(grid):
    return {"sky": jft.ShapeWithDtype((2, *grid.spatial.shape_yx), float)}


def _domain_keys(likelihood):
    return set(likelihood.response.domain) - {likelihood.filter}


def test_nicmos_evaluates_jits_and_differentiates(tmp_path):
    nicmos = _nicmos_entry(tmp_path, "n")
    grid = _grid(nicmos["path"])
    products = build_hst_likelihoods(
        _config(tmp_path, [nicmos]), grid, _sky_domain(grid)
    )

    assert list(products.sky_filter.bands) == [NIC2 + "n_cal"]
    # NICMOS MULTIACCUM gives a shot-noise model
    (likelihood,) = products.likelihoods
    assert isinstance(likelihood.builder, ShotNoiseGaussianLikelihoodBuilder)
    assert products.plotting.filter == list(products.sky_filter.bands)

    likelihood = products.likelihood
    x = jft.random_like(jax.random.PRNGKey(0), likelihood.domain)
    value = likelihood(x)
    assert np.isfinite(value)
    np.testing.assert_allclose(jax.jit(likelihood)(x), value, rtol=1e-6)
    gradient = jax.grad(likelihood)(x)
    assert all(np.all(np.isfinite(g)) for g in jax.tree.leaves(gradient))
    assert np.any(jax.tree.leaves(gradient["sky"])[0] != 0)


def test_wfpc2_likelihood_raises(tmp_path):
    nicmos = _nicmos_entry(tmp_path, "n")
    wfpc2 = dict(
        path=str(_wfpc2(tmp_path / "w_flt.fits")),
        camera="WF2",
        filter="F814W",
        chip=2,
        margin=2,
        psf=_kernel_file(tmp_path / "wf_psf.fits", scale=0.0996),
    )
    grid = _grid(nicmos["path"])
    match = "WFPC2 is not supported yet.*noise model.*charge diffusion"
    with pytest.raises(ValueError, match=match):
        build_hst_likelihoods(
            _config(tmp_path, [nicmos, wfpc2]), grid, _sky_domain(grid)
        )

    # the Python API, even with the exposure labelled as NICMOS
    loaded = load_hst_exposure(wfpc2["path"], grid, CURVE, margin=2, dq_bits=0, chip=2)
    kernel = HstPsfKernel(np.pad([[1.0]], 2), 0.0996, 1, True)
    exposure = Exposure(
        "hst", "NIC2", "F160W", "w", loaded.target_data, loaded.meta, CURVE, kernel
    )
    with pytest.raises(ValueError, match=match):
        build_exposure_likelihoods([exposure], grid, _sky_domain(grid), TELESCOPE)
    # and with a noise model the caller made up
    exposure = replace(exposure, noise=ShotNoise(a=25.0, b=1 / 7))
    with pytest.raises(ValueError, match=match):
        build_exposure_likelihoods([exposure], grid, _sky_domain(grid), TELESCOPE)


def test_one_filter_two_shapes_get_disjoint_keys_and_shared_weights(tmp_path):
    a, b = _nicmos_entry(tmp_path, "a"), _nicmos_entry(tmp_path, "b", margin=4)
    grid = _grid(a["path"])
    products = build_hst_likelihoods(_config(tmp_path, [a, b]), grid, _sky_domain(grid))

    first, second = products.likelihoods
    assert first.builder.data.shape != second.builder.data.shape
    keys = [_domain_keys(t) for t in products.likelihoods]
    assert keys[0] and keys[1] and not keys[0] & keys[1]
    for band, k in zip((NIC2 + "a_cal", NIC2 + "b_cal"), keys):
        assert {f"{band}_shift", f"{band}_target_calibration"} <= k
    bands = products.sky_filter.bands
    assert bands[NIC2 + "a_cal"] is bands[NIC2 + "b_cal"]
    x = jft.random_like(jax.random.PRNGKey(1), products.likelihood.domain)
    assert np.isfinite(products.likelihood(x))


def _exposure(tmp_path, subsample=1, psf=None):
    path = _nicmos(tmp_path / "n_cal.fits")
    grid = _grid(path)
    loaded = load_hst_exposure(
        path, grid, CURVE, margin=2, dq_bits=0, subsample=subsample
    )
    kernel = HstPsfKernel(np.pad([[1.0]], 2), 0.076, 1, False) if psf is None else psf
    exposure = Exposure(
        "hst",
        "NIC2",
        "F160W",
        "n",
        loaded.target_data,
        loaded.meta,
        CURVE,
        kernel,
        loaded.noise,
    )
    return exposure, grid


def test_psf_model_adds_only_its_keys(tmp_path):
    exposure, grid = _exposure(tmp_path)
    fixed = build_exposure_likelihoods([exposure], grid, _sky_domain(grid), TELESCOPE)

    width = jft.Model(
        lambda x: jnp.exp(x["psf_width"]),
        domain={"psf_width": jft.ShapeWithDtype((), float)},
    )
    centers = exposure.target_data.subsample_centers[0]
    model = PsfGaussianBroadening(
        jft.ShapeWithDtype((1, *centers.shape), float), exposure.psf.kernel[None], width
    )
    broadened = build_exposure_likelihoods(
        [replace(exposure, psf=model)], grid, _sky_domain(grid), TELESCOPE
    )

    fixed_keys = set(fixed.likelihood.domain)
    assert set(broadened.likelihood.domain) - fixed_keys == {"psf_width"}
    assert fixed_keys <= set(broadened.likelihood.domain)
    x = jft.random_like(jax.random.PRNGKey(2), broadened.likelihood.domain)
    assert np.isfinite(broadened.likelihood(x))


@pytest.mark.parametrize("kernel_subsample, data_subsample", [(1, 2), (2, 1)])
def test_kernel_sampling_must_match_the_subsample(
    tmp_path, kernel_subsample, data_subsample
):
    kernel = HstPsfKernel(np.ones((3, 3)) / 9, 0.076, kernel_subsample, False)
    exposure, grid = _exposure(tmp_path, data_subsample, kernel)
    with pytest.raises(ValueError, match="pixel response"):
        build_exposure_likelihoods([exposure], grid, _sky_domain(grid), TELESCOPE)


def test_duplicate_band_raises(tmp_path):
    exposure, grid = _exposure(tmp_path)
    with pytest.raises(ValueError, match="distinct band keys"):
        build_exposure_likelihoods(
            [exposure, exposure], grid, _sky_domain(grid), TELESCOPE
        )


def test_residuals_use_the_shot_noise_sigma(tmp_path):
    exposure, grid = _exposure(tmp_path)
    products = build_exposure_likelihoods(
        [exposure], grid, _sky_domain(grid), TELESCOPE
    )
    sky_domain = _sky_domain(grid)
    plots = SkyResiduals(
        residual_directory=str(tmp_path),
        sky_model=jft.Model(lambda x: {"sky": x["sky"]}, domain=sky_domain),
        sky_filter=products.sky_filter,
        residual_plotting_info=products.plotting,
    )
    x = jft.random_like(jax.random.PRNGKey(3), products.likelihood.domain).tree
    x["sky"] = jnp.abs(x["sky"]) * 1e3  # a positive sky, so b matters

    (band,) = products.sky_filter.bands
    mask = exposure.target_data.mask
    model = np.asarray(products.likelihoods[0].response(products.sky_filter(x) | x))
    noise = exposure.noise
    a, b, sky = (
        np.broadcast_to(v, mask.shape)[mask] for v in (noise.a, noise.b, noise.sky)
    )
    sigma = np.sqrt(a + b * np.maximum(model + sky, 0))
    assert not np.allclose(sigma, exposure.target_data.std[mask])
    expected = (exposure.target_data.data[mask] - model) / sigma
    np.testing.assert_allclose(plots.residuals(x)[band][mask], expected, rtol=1e-5)


def test_subsampled_kernel_at_its_subsample_builds(tmp_path):
    kernel = HstPsfKernel(np.ones((5, 5)) / 25, 0.038, 2, False)
    exposure, grid = _exposure(tmp_path, 2, kernel)
    products = build_exposure_likelihoods(
        [exposure], grid, _sky_domain(grid), TELESCOPE
    )
    x = jft.random_like(jax.random.PRNGKey(4), products.likelihood.domain)
    assert np.isfinite(products.likelihood(x))


def test_kernel_of_another_camera_raises(tmp_path):
    kernel = HstPsfKernel(np.pad([[1.0]], 2), 0.0455, 1, False)  # PC1 on NIC2
    exposure, grid = _exposure(tmp_path, psf=kernel)
    with pytest.raises(ValueError, match="arcsec pixels"):
        build_exposure_likelihoods([exposure], grid, _sky_domain(grid), TELESCOPE)


def test_another_throughput_for_one_filter_raises(tmp_path):
    a, b = _nicmos_entry(tmp_path, "a"), _nicmos_entry(tmp_path, "b")
    np.savetxt(tmp_path / "other.txt", np.c_[[1.4, 1.8], [1.0, 0.5]])
    b["throughput"] = str(tmp_path / "other.txt")
    grid = _grid(a["path"])
    with pytest.raises(ValueError, match="another throughput"):
        build_hst_likelihoods(_config(tmp_path, [a, b]), grid, _sky_domain(grid))


def test_per_band_shift_prior_matches_any_case(tmp_path):
    exposure, grid = _exposure(tmp_path)
    values = []
    for key in (None, exposure.band, exposure.band.lower()):
        telescope = deepcopy(TELESCOPE)
        priors = telescope["rotation_and_shift"]["correction_priors"]
        if key is not None:  # an offset fixed by sigma 0
            priors[key] = {
                "rotation": ["delta", 0.0, 0.1],
                "shift": ["normal", 0.05, 0.0],
            }
        products = build_exposure_likelihoods(
            [exposure], grid, _sky_domain(grid), telescope
        )
        x = jft.random_like(jax.random.PRNGKey(5), products.likelihood.domain)
        values.append(float(products.likelihood(x)))
    default, exact, lower = values
    assert exact == lower != default


@pytest.mark.parametrize("section", ["zero_flux", "calibration", "correction_priors"])
def test_prior_key_of_no_band_raises(tmp_path, section):
    exposure, grid = _exposure(tmp_path)
    telescope = deepcopy(TELESCOPE)
    priors = telescope.get(section) or telescope["rotation_and_shift"][section]
    priors["f160w"] = priors["default"]  # the JWST-style filter key
    with pytest.raises(ValueError, match="match no band"):
        build_exposure_likelihoods([exposure], grid, _sky_domain(grid), telescope)


def test_cropping_keeps_the_likelihood_on_a_coarse_grid(tmp_path):
    """Sky pixels of 4 chip pixels: the cutout must still hold every pixel the
    sky-grid mask keeps, so cut and whole chip give the same likelihood."""
    path = _nicmos(tmp_path / "n_cal.fits")
    with fits.open(path) as hdul:
        center = WCS(hdul["SCI", 1].header, hdul).pixel_to_world(25.3, 19.6)
    fov = [3 * 4 * s for s in NIC2_SCALE] * u.arcsec
    grid = Grid(WcsAstropy(center, (3, 3), fov, 20 * u.deg), SPECTRAL)
    profile = np.exp(-0.5 * (np.arange(5) - 2.0) ** 2)
    kernel = HstPsfKernel(
        np.outer(profile, profile) / profile.sum() ** 2, 0.076, 1, False
    )
    # wrap: the sky beyond the grid is not zero, so the margin matters
    rotation_and_shift = TELESCOPE["rotation_and_shift"] | {
        "linear": {"order": 1, "mode": "wrap"}
    }
    telescope = TELESCOPE | {"rotation_and_shift": rotation_and_shift}

    def build(margin):
        loaded = load_hst_exposure(path, grid, CURVE, margin=margin, dq_bits=0)
        exposure = Exposure(
            "hst",
            "NIC2",
            "F160W",
            "n",
            loaded.target_data,
            loaded.meta,
            CURVE,
            kernel,
            loaded.noise,
        )
        return build_exposure_likelihoods(
            [exposure], grid, _sky_domain(grid), telescope
        )

    cut, whole = build(2), build(1000)
    (cut_builder,), (whole_builder,) = (
        [t.builder for t in p.likelihoods] for p in (cut, whole)
    )
    assert whole_builder.data.shape == (1, *SHAPE)
    assert cut_builder.data.size < whole_builder.data.size
    assert cut_builder.mask.sum() == whole_builder.mask.sum() > 0

    x = jft.random_like(jax.random.PRNGKey(6), whole.likelihood.domain)
    x = x.tree if isinstance(x, jft.Vector) else x
    x["sky"] = 1e3 * jnp.abs(x["sky"])
    np.testing.assert_allclose(cut.likelihood(x), whole.likelihood(x), rtol=1e-12)


def _evaluated(x):
    """1 for an evaluation on arrays, 0 for a shape trace by jft.Model."""
    return not isinstance(jax.tree.leaves(x)[0], jax.core.Tracer)


def _two_band_residuals(tmp_path):
    exposure, grid = _exposure(tmp_path)
    exposures = [exposure, replace(exposure, exposure="m")]
    products = build_exposure_likelihoods(exposures, grid, _sky_domain(grid), TELESCOPE)
    calls = {"sky": 0}

    def sky_model(x):
        calls["sky"] += _evaluated(x)
        return {"sky": x["sky"]}

    plots = SkyResiduals(
        residual_directory=str(tmp_path),
        sky_model=jft.Model(sky_model, domain=_sky_domain(grid)),
        sky_filter=products.sky_filter,
        residual_plotting_info=products.plotting,
    )
    return exposures, products, plots, calls


def test_residual_samples_evaluate_sky_and_model_once_each(tmp_path):
    exposures, products, plots, calls = _two_band_residuals(tmp_path)
    responses = [t.builder.response for t in products.likelihoods]
    for t, response in zip(products.likelihoods, responses):
        calls[t.filter] = 0

        def counted(x, response=response, band=t.filter):
            calls[band] += _evaluated(x)
            return response(x)

        t.builder.response = jft.Model(counted, domain=response.domain)

    pos = jft.random_like(jax.random.PRNGKey(7), products.likelihood.domain)
    pos = jft.Vector(pos.tree if isinstance(pos, jft.Vector) else pos)
    pos.tree["sky"] = jnp.abs(pos.tree["sky"]) * 1e3  # a positive sky, so b matters
    delta = jax.tree.map(
        lambda v: 0.1 * v,
        jft.random_like(jax.random.PRNGKey(8), products.likelihood.domain),
    )
    delta = jft.Vector(delta.tree if isinstance(delta, jft.Vector) else delta)
    samples = jft.Samples(
        pos=pos, samples=jax.tree.map(lambda d: jnp.stack([d, -d]), delta)
    )
    residuals = plots.residuals(samples)

    # one sky per sample, one response per sample and exposure
    assert calls == {"sky": 2} | {t.filter: 2 for t in products.likelihoods}
    for e, t, response in zip(exposures, products.likelihoods, responses):
        mask = e.target_data.mask
        a, b, sky = (
            np.broadcast_to(v, mask.shape)[mask]
            for v in (e.noise.a, e.noise.b, e.noise.sky)
        )
        models, sigmas = [], []
        for s in samples:
            s = s.tree if isinstance(s, jft.Vector) else s
            model = np.asarray(response(products.sky_filter(s) | s))
            models.append(model)
            sigmas.append(np.sqrt(a + b * np.maximum(model + sky, 0)))
        expected = (e.target_data.data[mask] - np.mean(models, 0)) / np.mean(sigmas, 0)
        np.testing.assert_allclose(residuals[t.filter][mask], expected, rtol=1e-10)


def test_residual_pages_bound_the_figure(tmp_path, monkeypatch):
    _, products, plots, _ = _two_band_residuals(tmp_path)
    x = jft.random_like(jax.random.PRNGKey(9), products.likelihood.domain)
    x = x.tree if isinstance(x, jft.Vector) else x

    plots(x, SimpleNamespace(nit=3))  # 2 rows of 4 panels fit one page
    assert sorted(p.name for p in tmp_path.glob("*.png")) == ["03.png"]
    assert matplotlib.image.imread(tmp_path / "03.png").shape[:2] == (1800, 3600)

    monkeypatch.setattr(residual_plots, "MAX_PANELS_PER_PAGE", 4)  # one row
    plots(x, SimpleNamespace(nit=4))
    pages = sorted(p.name for p in tmp_path.glob("04*.png"))
    assert pages == ["04_00.png", "04_01.png"]
    for page in pages:
        assert matplotlib.image.imread(tmp_path / page).shape[:2] == (900, 3600)
