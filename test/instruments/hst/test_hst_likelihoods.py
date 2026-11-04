"""build_hst_likelihoods and the shared per-exposure builder on the synthetic
NICMOS and WFPC2 files of the loader tests."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import nifty.re as jft
import numpy as np
import pytest
from astropy.io import fits

from jubik.instruments.hst.hst_data import load_hst_exposure
from jubik.instruments.hst.hst_likelihoods import build_hst_likelihoods
from jubik.instruments.hst.tinytim import HstPsfKernel
from jubik.instruments.jwst.exposure_likelihoods import (
    Exposure,
    build_exposure_likelihoods,
)
from jubik.instruments.jwst.likelihood.likelihood import (
    GaussianLikelihoodBuilder,
    ShotNoiseGaussianLikelihoodBuilder,
)
from jubik.instruments.jwst.plotting.residuals import SkyResiduals
from jubik.instruments.jwst.psf.psf_operator import PsfGaussianBroadening
from test_hst_data import CURVE, _grid, _nicmos, _wfpc2

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
WF = "hst_WF2_F814W_w"


def _kernel_file(path, subsample=1, size=5, scale=0.076):
    kernel = np.exp(
        -0.5 * (np.arange(size * subsample) - (size * subsample - 1) / 2) ** 2
    )
    hdu = fits.PrimaryHDU(np.outer(kernel, kernel))
    hdu.header.update(PIXSCALE=scale / subsample, SUBSAMPL=subsample, CHGDIFF=False)
    hdu.writeto(path)
    return str(path)


def _config(tmp_path, entries, **telescope):
    np.savetxt(tmp_path / "curve.txt", np.c_[CURVE.wavelength.value, CURVE.throughput])
    psf = _kernel_file(tmp_path / "psf.fits")
    common = dict(throughput=str(tmp_path / "curve.txt"), psf=psf, dq_bits=0)
    return {
        "files": {"exposures": [common | e for e in entries]},
        "telescope": TELESCOPE | telescope,
    }


def _nicmos_entry(tmp_path, name, margin=2):
    path = _nicmos(tmp_path / f"{name}_cal.fits")
    return dict(path=str(path), camera="NIC2", filter="F160W", margin=margin)


def _sky_domain(grid):
    return {"sky": jft.ShapeWithDtype((2, *grid.spatial.shape_yx), float)}


def _domain_keys(likelihood):
    return set(likelihood.response.domain) - {likelihood.filter}


def test_nicmos_and_wfpc2_evaluate_jit_and_differentiate(tmp_path):
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
    products = build_hst_likelihoods(
        _config(tmp_path, [nicmos, wfpc2]), grid, _sky_domain(grid)
    )

    assert list(products.sky_filter.bands) == [NIC2 + "n_cal", WF + "_flt"]
    builders = [t.builder for t in products.likelihoods]
    # NICMOS MULTIACCUM gives a shot-noise model, WFPC2 none
    assert isinstance(builders[0], ShotNoiseGaussianLikelihoodBuilder)
    assert type(builders[1]) is GaussianLikelihoodBuilder
    assert products.plotting.filter == list(products.sky_filter.bands)

    likelihood = products.likelihood
    x = jft.random_like(jax.random.PRNGKey(0), likelihood.domain)
    value = likelihood(x)
    assert np.isfinite(value)
    np.testing.assert_allclose(jax.jit(likelihood)(x), value, rtol=1e-6)
    gradient = jax.grad(likelihood)(x)
    assert all(np.all(np.isfinite(g)) for g in jax.tree.leaves(gradient))
    assert np.any(jax.tree.leaves(gradient["sky"])[0] != 0)


def test_shot_noise_false_gives_the_fixed_sigma_gaussian(tmp_path):
    nicmos = _nicmos_entry(tmp_path, "n")
    grid = _grid(nicmos["path"])
    cfg = _config(tmp_path, [nicmos], shot_noise=False)
    products = build_hst_likelihoods(cfg, grid, _sky_domain(grid))
    assert type(products.likelihoods[0].builder) is GaussianLikelihoodBuilder


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
