"""SkyFilter: partial channel weights, coverage policy, one-hot limit."""

import logging

import jax.numpy as jnp
import nifty.re as jft
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from jubik.color import Color
from jubik.grid import Grid
from jubik.instruments.jwst.data.jwst_information import JWST_FILTERS
from jubik.sky_filter import MAX_MISSING, SKY_KEY, SkyFilter, Transmission, band_weights
from jubik.wcs.wcs_astropy import WcsAstropy

NY = 8
# rising edge 3.9 to 4.1 um, flat to 5.0 um
RAMP = Transmission(np.array([3.9, 4.1, 5.0]), np.array([0.0, 1.0, 1.0]))


def _grid(spectral: Color) -> Grid:
    spatial = WcsAstropy(
        center=SkyCoord(ra=64.665 * u.deg, dec=-47.865 * u.deg),
        shape=(NY, NY),
        fov=(1 * u.arcsec, 1 * u.arcsec),
    )
    return Grid(spatial=spatial, spectral=spectral)


def _spectral(bounds) -> Color:
    return Color(np.asarray(bounds, float) * u.um)


def _edges_grid(edges_um) -> Grid:
    return _grid(_spectral(edges_um))


def _sky(n_ch: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n_ch, NY, NY)).astype(np.float32)


def _ramp_integral(lo, hi):
    """Analytic integral of RAMP(lam) / lam over [lo, hi]."""
    lo, hi = max(lo, 3.9), min(hi, 5.0)
    if hi <= lo:
        return 0.0
    # ramp part: T = (lam - 3.9) / 0.2 on [3.9, 4.1]
    a, b = lo, min(hi, 4.1)
    ramp = ((b - a) - 3.9 * np.log(b / a)) / 0.2 if b > a else 0.0
    # flat part: T = 1 on [4.1, 5.0]
    a, b = max(lo, 4.1), hi
    flat = np.log(b / a) if b > a else 0.0
    return ramp + flat


@pytest.mark.parametrize(
    "bounds",
    [
        [[3.5, 4.3], [4.3, 5.2]],  # case 1: edge cuts one channel boundary
        [[3.5, 4.0], [4.0, 4.6], [4.6, 5.2]],  # case 2: edge inside the first channel
    ],
)
def test_soft_edge_gives_partial_weights(bounds):
    bw = band_weights(_spectral(bounds), RAMP)
    expected = np.array([_ramp_integral(lo, hi) for lo, hi in bounds])
    expected /= expected.sum()
    assert bw.sl == slice(0, len(bounds))
    np.testing.assert_allclose(bw.w, expected, rtol=1e-5)
    np.testing.assert_allclose(bw.w.sum(), 1.0, rtol=1e-12)


def test_flat_sky_is_preserved():
    grid = _edges_grid(np.linspace(1.0, 2.0, 11))
    bands = {
        "box": Transmission.tophat(1.23, 1.71),
        "ramp": Transmission(np.array([1.1, 1.5, 1.9]), np.array([0.2, 1.0, 0.5])),
    }
    sky_filter = SkyFilter(grid, bands)
    for c in (7.5, -2.0):
        out = sky_filter({SKY_KEY: jnp.full((10, NY, NY), c, jnp.float32)})
        for key, y in out.items():
            assert y.shape == (NY, NY)
            np.testing.assert_allclose(np.asarray(y), c, rtol=1e-6, err_msg=key)


def test_tophat_weights_are_log_ratios():
    bw = band_weights(
        _spectral([[1.0, 1.5], [1.5, 2.0], [2.0, 3.0]]), Transmission.tophat(1.2, 2.4)
    )
    expected = np.log([1.5 / 1.2, 2.0 / 1.5, 2.4 / 2.0])
    np.testing.assert_allclose(bw.w, expected / expected.sum(), rtol=1e-6)


def _warnings(fn, caplog):
    # nifty's logger does not propagate, so caplog listens on it directly
    jft.logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=jft.logger.name):
            return fn()
    finally:
        jft.logger.removeHandler(caplog.handler)


def test_missing_passband_raises_above_max_missing():
    with pytest.raises(ValueError, match="biased.*outside the sky"):
        band_weights(_spectral([[4.0, 5.2]]), RAMP, name="biased")
    with pytest.raises(ValueError, match="gap"):
        band_weights(_spectral([[3.5, 4.0], [4.3, 5.2]]), RAMP, name="gap")


def test_small_missing_passband_warns(caplog):
    # overhang holding about half of MAX_MISSING
    hi = 2.0 * (1 + 0.5 * MAX_MISSING * np.log(2.0 / 1.5))
    spectral = _spectral([[1.0, 1.5], [1.5, 2.0]])
    bw = _warnings(
        lambda: band_weights(spectral, Transmission.tophat(1.5, hi), name="wing"),
        caplog,
    )
    assert bw.sl == slice(1, 2)
    np.testing.assert_allclose(bw.w, [1.0])
    assert "wing" in caplog.text and "renormalised" in caplog.text
    with pytest.raises(ValueError, match="wing"):
        band_weights(
            spectral, Transmission.tophat(1.5, hi), max_missing=0.0, name="wing"
        )


def test_full_coverage_does_not_warn(caplog):
    spectral = _spectral([[1.0, 1.5], [1.5, 2.0]])
    _warnings(
        lambda: band_weights(spectral, Transmission.tophat(1.5, 2.0), max_missing=0.0),
        caplog,
    )
    inside = Transmission(np.array([1.2, 1.4, 1.9]), np.array([0.0, 1.0, 1.0]))
    _warnings(lambda: band_weights(spectral, inside, max_missing=0.0), caplog)
    assert caplog.text == ""


def test_one_channel_per_filter_is_one_hot():
    names = ["F150W", "F277W", "F444W"]
    bounds = np.array([JWST_FILTERS[n][3:5] for n in names])
    grid = _grid(_spectral(bounds))
    sky_filter = SkyFilter(
        grid, {n: Transmission.tophat(*JWST_FILTERS[n][3:5]) for n in names}
    )
    sky = _sky(len(names), seed=1)
    out = sky_filter({SKY_KEY: jnp.asarray(sky)})
    assert list(out) == names
    for index, n in enumerate(names):
        assert sky_filter.weights[n].sl == slice(index, index + 1)
        np.testing.assert_allclose(sky_filter.weights[n].w, [1.0])
        np.testing.assert_allclose(np.asarray(out[n]), sky[index], atol=1e-6)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_domain_and_dtype(dtype):
    grid = _edges_grid(np.linspace(1.0, 2.0, 11))
    sky_filter = SkyFilter(grid, {"f": Transmission.tophat(1.2, 1.8)}, dtype=dtype)
    assert sky_filter.domain[SKY_KEY].shape == (10, NY, NY)
    assert sky_filter.domain[SKY_KEY].dtype == dtype
    out = sky_filter({SKY_KEY: jnp.asarray(_sky(10), dtype=dtype)})
    assert out["f"].shape == (NY, NY) and out["f"].dtype == dtype


def test_transmission_validation():
    with pytest.raises(ValueError, match="ascending"):
        Transmission(np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="same length"):
        Transmission(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="non-negative"):
        Transmission(np.array([1.0, 2.0]), np.array([0.0, 0.0]))
