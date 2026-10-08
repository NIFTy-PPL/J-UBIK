"""SkyFilter: weighted sums per band, planes and cubes, contract validation."""

import jax.numpy as jnp
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from jubik.color import Color
from jubik.grid import Grid
from jubik.sky_filter import SKY_KEY, FilterWeights, SkyFilter
from jubik.wcs.wcs_astropy import WcsAstropy

NY = 8
N_CH = 10


def _grid(n_ch: int = N_CH) -> Grid:
    spatial = WcsAstropy(
        center=SkyCoord(ra=64.665 * u.deg, dec=-47.865 * u.deg),
        shape=(NY, NY),
        fov=(1 * u.arcsec, 1 * u.arcsec),
    )
    return Grid(spatial=spatial, spectral=Color(np.linspace(1.0, 2.0, n_ch + 1) * u.um))


def _sky(n_ch: int = N_CH, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n_ch, NY, NY)).astype(np.float32)


def test_plane_is_the_weighted_sum():
    band = FilterWeights(slice(2, 5), np.array([0.2, 0.5, 0.3]))
    sky = _sky()
    out = SkyFilter(_grid(), {"f": band})({SKY_KEY: jnp.asarray(sky)})
    assert list(out) == ["f"] and out["f"].shape == (NY, NY)
    expected = 0.2 * sky[2] + 0.5 * sky[3] + 0.3 * sky[4]
    np.testing.assert_allclose(np.asarray(out["f"]), expected, rtol=1e-5)


def test_cube_is_one_row_per_output_bin():
    W = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]])
    band = FilterWeights(slice(4, 7), W)
    sky = _sky()
    out = SkyFilter(_grid(), {"ifu": band})({SKY_KEY: jnp.asarray(sky)})["ifu"]
    assert out.shape == (2, NY, NY)
    np.testing.assert_allclose(
        np.asarray(out), np.tensordot(W, sky[4:7], axes=(1, 0)), rtol=1e-5
    )


def test_identity_rows_return_the_sub_cube():
    band = FilterWeights(slice(2, 6), np.eye(4))
    sky = _sky()
    out = SkyFilter(_grid(), {"ifu": band})({SKY_KEY: jnp.asarray(sky)})["ifu"]
    np.testing.assert_array_equal(np.asarray(out), sky[2:6])


def test_one_hot_returns_the_channel():
    bands = {f"b{i}": FilterWeights(slice(i, i + 1), np.ones(1)) for i in range(3)}
    sky = _sky()
    out = SkyFilter(_grid(), bands)({SKY_KEY: jnp.asarray(sky)})
    assert list(out) == list(bands)
    for i, name in enumerate(bands):
        np.testing.assert_array_equal(np.asarray(out[name]), sky[i])


def test_one_row_cube_and_plane_agree():
    w = np.array([0.4, 0.6])
    bands = {
        "plane": FilterWeights(slice(0, 2), w),
        "cube": FilterWeights(slice(0, 2), w[None]),
    }
    out = SkyFilter(_grid(), bands)({SKY_KEY: jnp.asarray(_sky())})
    assert out["plane"].shape == (NY, NY) and out["cube"].shape == (1, NY, NY)
    np.testing.assert_allclose(np.asarray(out["cube"][0]), np.asarray(out["plane"]))


def test_normalised_weights_preserve_a_flat_sky():
    bands = {
        "f": FilterWeights(slice(1, 4), np.array([0.1, 0.6, 0.3])),
        "ifu": FilterWeights(slice(0, 4), np.full((3, 4), 0.25)),
    }
    sky_filter = SkyFilter(_grid(), bands)
    for c in (7.5, -2.0):
        out = sky_filter({SKY_KEY: jnp.full((N_CH, NY, NY), c, jnp.float32)})
        for name, y in out.items():
            np.testing.assert_allclose(np.asarray(y), c, rtol=1e-6, err_msg=name)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_domain_and_dtype(dtype):
    bands = {
        "f": FilterWeights(slice(0, 2), np.ones(2) / 2),
        "i": FilterWeights(slice(0, 3), np.eye(3)),
    }
    sky_filter = SkyFilter(_grid(), bands, dtype=dtype)
    assert sky_filter.domain[SKY_KEY].shape == (N_CH, NY, NY)
    assert sky_filter.domain[SKY_KEY].dtype == dtype
    assert sky_filter.target["f"].shape == (NY, NY) and sky_filter.target[
        "i"
    ].shape == (3, NY, NY)
    out = sky_filter({SKY_KEY: jnp.asarray(_sky(), dtype=dtype)})
    assert out["f"].dtype == dtype and out["i"].dtype == dtype


def test_channels_beyond_the_sky_raise():
    with pytest.raises(ValueError, match="beyond"):
        SkyFilter(_grid(), {"f": FilterWeights(slice(8, 11), np.ones(3) / 3)})


def test_filter_weights_validation():
    with pytest.raises(ValueError, match="slice"):
        FilterWeights(slice(None, 3), np.ones(3))
    with pytest.raises(ValueError, match="slice"):
        FilterWeights(slice(3, 3), np.ones(0))
    with pytest.raises(ValueError, match="slice"):
        FilterWeights(slice(0, 4, 2), np.ones(2))
    with pytest.raises(ValueError, match="n_channels = 3"):
        FilterWeights(slice(0, 3), np.ones(2))
    with pytest.raises(ValueError, match="n_channels = 3"):
        FilterWeights(slice(0, 3), np.ones((2, 2)))
    with pytest.raises(ValueError, match=r"\(n_channels,\) or"):
        FilterWeights(slice(0, 3), np.ones((1, 1, 3)))
    with pytest.raises(ValueError, match="finite"):
        FilterWeights(slice(0, 2), np.array([np.nan, 1.0]))
    band = FilterWeights(slice(np.int64(1), np.int64(3)), [0.5, 0.5])
    assert band.channels == slice(1, 3) and band.weights.dtype == float
