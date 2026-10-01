"""SkyProjection: band selection, normalisation, coverage checks, one-hot limit."""

import jax.numpy as jnp
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from jubik.color import Color
from jubik.grid import Grid
from jubik.instruments.jwst.data.jwst_information import JWST_FILTERS
from jubik.sky_projection import (
    MAX_MISSING,
    SKY_KEY,
    FilterBand,
    IfuBand,
    SkyProjection,
    Transmission,
    filter_selection,
)
from jubik.wcs.wcs_astropy import WcsAstropy

NY = 8


def _grid(spectral: Color) -> Grid:
    spatial = WcsAstropy(
        center=SkyCoord(ra=64.665 * u.deg, dec=-47.865 * u.deg),
        shape=(NY, NY),
        fov=(1 * u.arcsec, 1 * u.arcsec),
    )
    return Grid(spatial=spatial, spectral=spectral)


def _edges_grid(edges_um) -> Grid:
    return _grid(Color(np.asarray(edges_um, float) * u.um))


def _sky(n_ch: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n_ch, NY, NY)).astype(np.float32)


def test_flat_sky_is_preserved():
    grid = _edges_grid(np.linspace(1.0, 2.0, 11))
    bands = [
        FilterBand("box", Transmission.tophat(1.23, 1.71)),
        FilterBand(
            "ramp", Transmission(np.array([1.1, 1.5, 1.9]), np.array([0.2, 1.0, 0.5]))
        ),
        IfuBand("ifu", 1.3, 1.6),
    ]
    proj = SkyProjection(grid, bands)
    for c in (7.5, -2.0):
        out = proj({SKY_KEY: jnp.full((10, NY, NY), c, jnp.float32)})
        for key, y in out.items():
            np.testing.assert_allclose(np.asarray(y), c, rtol=1e-6, err_msg=key)


def test_missing_passband_beyond_last_edge():
    spectral = Color(np.linspace(1.0, 2.0, 11) * u.um)
    with pytest.raises(ValueError, match="box"):
        filter_selection(spectral, FilterBand("box", Transmission.tophat(1.5, 2.1)))
    # 1/g-weighted overhang just under MAX_MISSING still passes
    hi = 2.0 + 0.5 * MAX_MISSING * 0.5
    sel = filter_selection(spectral, FilterBand("box", Transmission.tophat(1.5, hi)))
    assert sel.plane and sel.W.shape == (1, 5)
    np.testing.assert_allclose(sel.W.sum(), 1.0, rtol=1e-12)


def test_gap_inside_passband_raises():
    bounds = np.array([[1.0, 1.2], [1.2, 1.4], [1.5, 1.7], [1.7, 1.9]])
    spectral = Color(bounds * u.um)
    with pytest.raises(ValueError, match="outside the sky"):
        filter_selection(spectral, FilterBand("gap", Transmission.tophat(1.1, 1.8)))
    sel = filter_selection(spectral, FilterBand("ok", Transmission.tophat(1.5, 1.85)))
    assert sel.sl == slice(2, 4)


def test_ifu_band_selects_overlapping_channels():
    grid = _edges_grid(np.linspace(1.0, 2.0, 11))
    proj = SkyProjection(grid, [IfuBand("part", 1.25, 1.52), IfuBand("all")])
    part, every = proj.selections["part"], proj.selections["all"]
    assert part.sl == slice(2, 6)
    np.testing.assert_array_equal(part.W, np.eye(4))
    assert proj.band_grid("part").spectral.center.size == part.W.shape[0]
    assert every.sl == slice(0, 10) and every.W.shape == (10, 10)
    sky = _sky(10)
    out = proj({SKY_KEY: jnp.asarray(sky)})
    np.testing.assert_array_equal(np.asarray(out["part"]), sky[2:6])
    np.testing.assert_array_equal(np.asarray(out["all"]), sky)
    with pytest.raises(ValueError, match="no sky channel"):
        SkyProjection(grid, [IfuBand("none", 2.5, 3.0)])


def test_one_channel_per_filter_is_one_hot():
    names = ["F150W", "F277W", "F444W"]
    bounds = np.array([JWST_FILTERS[n][3:5] for n in names])
    grid = _grid(Color(bounds * u.um))
    proj = SkyProjection(
        grid, [FilterBand(n, Transmission.tophat(*JWST_FILTERS[n][3:5])) for n in names]
    )
    sky = _sky(len(names), seed=1)
    out = proj({SKY_KEY: jnp.asarray(sky)})
    assert list(out) == names
    for index, n in enumerate(names):
        assert proj.selections[n].sl == slice(index, index + 1)
        np.testing.assert_allclose(proj.selections[n].W, [[1.0]])
        np.testing.assert_allclose(np.asarray(out[n]), sky[index], atol=1e-6)


def test_duplicate_band_keys_raise():
    grid = _edges_grid(np.linspace(1.0, 2.0, 11))
    band = FilterBand("f", Transmission.tophat(1.2, 1.8))
    with pytest.raises(ValueError, match="duplicate"):
        SkyProjection(grid, [band, band])


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_output_shapes_and_dtype(dtype):
    grid = _edges_grid(np.linspace(1.0, 2.0, 11))
    proj = SkyProjection(
        grid,
        [FilterBand("f", Transmission.tophat(1.2, 1.8)), IfuBand("i", 1.0, 1.3)],
        dtype=dtype,
    )
    assert proj.domain[SKY_KEY].shape == (10, NY, NY)
    assert proj.domain[SKY_KEY].dtype == dtype
    out = proj({SKY_KEY: jnp.asarray(_sky(10), dtype=dtype)})
    assert out["f"].shape == (NY, NY) and out["i"].shape == (3, NY, NY)
    assert out["f"].dtype == dtype and out["i"].dtype == dtype
