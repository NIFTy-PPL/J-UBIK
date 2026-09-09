"""Smoke test for `build_jwst_likelihoods`.

Runs the full chain, config parsing, preload, load, response, likelihood, on a
fake observation. Two production seams are patched:

- The `JwstData` name in the preloader and the three loaders, rebound to
  `FakeJwstData`, which is a `JwstData` with an astropy WCS and in-memory arrays
  instead of a `jwst` datamodel.
- `build_webb_psf` in `psf/jwst_kernel.py`, replaced by a delta kernel so that
  stpsf is never imported. `load_psf_kernel` still writes and reads the `.npy`.
"""

from types import SimpleNamespace

import jax
import nifty.re as jft
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from jubik.color import Color
from jubik.grid import Grid
from jubik.instruments.jwst.data.jwst_data import DataMetaInformation, JwstData
from jubik.instruments.jwst.data.jwst_information import (
    get_dvol,
    get_pixel_distance,
)
from jubik.instruments.jwst.jwst_likelihoods import build_jwst_likelihoods
from jubik.instruments.jwst.psf import jwst_kernel
from jubik.wcs import WcsAstropy

FILTER = "F444W"
CENTER = SkyCoord(ra=10.05, dec=-4.95, unit="deg")
DATA_SHAPE = (64, 64)
NAN_PIXEL = (32, 30)
GRID_SHAPE = (16, 16)
GRID_FOV = [1.6, 1.6] * u.arcsec
SKY_DOMAIN = {"sky": jft.ShapeWithDtype((1,) + GRID_SHAPE, float)}

JWST_DATA_SITES = (
    "jubik.instruments.jwst.data.preloader.preloader.JwstData",
    "jubik.instruments.jwst.data.loader.data_loader.JwstData",
    "jubik.instruments.jwst.data.loader.target_loader.JwstData",
    "jubik.instruments.jwst.data.loader.stars_loader.JwstData",
)


class FakeJwstData(JwstData):
    """`JwstData` backed by arrays and an astropy WCS instead of a datamodel."""

    def __init__(self, filepath: str):
        pixel_scale = get_pixel_distance(FILTER)
        rng = np.random.default_rng(abs(hash(str(filepath))) % 2**32)
        data = rng.normal(size=DATA_SHAPE)
        data[NAN_PIXEL] = np.nan
        self.dm = SimpleNamespace(
            data=data,
            err=np.ones(DATA_SHAPE),
            meta=SimpleNamespace(
                date="2023-01-01T00:00:00.000",
                pointing=SimpleNamespace(ra_v1=CENTER.ra.deg, dec_v1=CENTER.dec.deg),
                wcsinfo=SimpleNamespace(ra_ref=CENTER.ra.deg, dec_ref=CENTER.dec.deg),
            ),
        )
        self.wcs = WcsAstropy(
            center=CENTER,
            shape=DATA_SHAPE,
            fov=u.Quantity([pixel_scale * s for s in DATA_SHAPE]),
        )
        self.shape = DATA_SHAPE
        self.filter = FILTER
        self.camera = "NIRCAM"
        self.meta = DataMetaInformation(
            unit=u.Unit("MJy/sr"),
            dvol=get_dvol(FILTER),
            pixel_scale=pixel_scale,
            color=self.pivot_wavelength,
        )


def fake_build_webb_psf(camera, filter, center_pixel, webbpsf_path, subsample, *a, **k):
    size = 5 * subsample
    psf = np.zeros((size, size))
    psf[size // 2, size // 2] = 1.0
    return psf


@pytest.fixture
def patched_seams(monkeypatch):
    for site in JWST_DATA_SITES:
        monkeypatch.setattr(site, FakeJwstData)
    monkeypatch.setattr(jwst_kernel, "build_webb_psf", fake_build_webb_psf)


def make_grid():
    spatial = WcsAstropy(center=CENTER, shape=GRID_SHAPE, fov=GRID_FOV)
    return Grid(spatial=spatial, spectral=Color([3.9, 5.0] * u.um))


def make_config(
    tmp_path, n_files=1, gaia=False, variable_covariance=False, zero_flux=False
):
    files = [str(tmp_path / f"fake_{ii}.fits") for ii in range(n_files)]
    telescope = {
        "psf": {
            "webbpsf_path": "",
            "psf_library": str(tmp_path / "psf_library"),
            "psf_arcsec_extension": 0.4,
        },
        "target": {"subsample": 2},
        "rotation_and_shift": {
            "linear": {"order": 1, "mode": "constant"},
            "correction_priors": {
                "model": "shift",
                "shift_unit": "arcsec",
                "rotation_unit": "deg",
                "default": {
                    "rotation": ["delta", 0.0, 0.1],
                    "shift": ["normal", 0.0, 0.1],
                },
            },
        },
    }
    if gaia:
        (tmp_path / "gaia").mkdir()
        telescope["gaia_alignment"] = {
            "fov": "0.5arcsec",
            "subsample": 2,
            "star_light": ["lognormal", 1.0, 1.0],
            "library_path": str(tmp_path / "gaia"),
        }
    if zero_flux:
        telescope["zero_flux"] = {"default": ["lognormal", 0.9, 4]}
    if variable_covariance:
        telescope["variable_covariance"] = {
            "additive_std_value": {
                "shape_type": "pixel",
                "distribution": ["invgamma", 4, 0.01, 0.0],
            }
        }
    return {"files": {"filter": {FILTER.lower(): files}}, "telescope": telescope}


def evaluate(likelihood):
    x = jft.random_like(jax.random.PRNGKey(0), likelihood.domain)
    return float(likelihood(x))


def test_minimal_config_builds_and_evaluates(patched_seams, tmp_path):
    cfg = make_config(tmp_path)
    products = build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN)

    assert products.alignment is None
    assert len(products.target.likelihoods) == 1
    target = products.target.likelihoods[0]
    assert target.filter == FILTER.lower()

    builder = target.builder
    assert builder.data.shape[0] == 1
    assert builder.data.shape == builder.mask.shape == builder.std.shape
    assert builder.mask.sum() > 0
    assert not np.isnan(builder.data[builder.mask]).any()

    likelihood = products.target.likelihood
    assert "sky" in likelihood.domain
    assert np.isfinite(evaluate(likelihood))

    saved = list((tmp_path / "psf_library").glob("*.npy"))
    assert len(saved) == 1
    assert "nircam_f444w" in saved[0].name and saved[0].name.endswith("sub2.npy")


def test_two_files_share_bounds(patched_seams, tmp_path):
    cfg = make_config(tmp_path, n_files=2)
    products = build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN)

    builder = products.target.likelihoods[0].builder
    assert builder.data.shape[0] == 2
    assert not np.allclose(builder.data[0], builder.data[1], equal_nan=True)
    assert np.isfinite(evaluate(products.target.likelihood))


@pytest.mark.parametrize("zero_flux", [True, False])
def test_gaia_and_variable_covariance(patched_seams, fake_gaia, tmp_path, zero_flux):
    calls, table = fake_gaia
    cfg = make_config(
        tmp_path, gaia=True, variable_covariance=True, zero_flux=zero_flux
    )
    products = build_jwst_likelihoods(cfg, make_grid(), SKY_DOMAIN)

    assert len(calls) == 1
    assert products.alignment is not None
    assert len(products.alignment.likelihood.likelihoods) == 1

    target = products.target.likelihood
    minimal = build_jwst_likelihoods(make_config(tmp_path), make_grid(), SKY_DOMAIN)
    extra_keys = set(target.domain.tree) - set(minimal.target.likelihood.domain.tree)
    assert extra_keys, "variable covariance should add parameters"
    assert any("zero_flux" in k for k in extra_keys) == zero_flux
    assert np.isfinite(evaluate(target))

    alignment = products.alignment.likelihood.likelihood
    keys = set(alignment.domain.tree)
    for star_id in table["SOURCE_ID"]:
        assert f"{FILTER.lower()}_{star_id}_brightness" in keys
    assert any("zero_flux" in k for k in keys) == zero_flux
    assert np.isfinite(evaluate(alignment))
