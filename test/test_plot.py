import matplotlib
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import jax.numpy as jnp
import pytest

import jubik as ju
from jubik.plot import (
    _get_color_limits,
    _get_n_rows_from_n_samples,
    _get_nside_from_npix,
)


matplotlib.use("Agg")


def _assert_written_image(path):
    assert path.exists()
    assert path.stat().st_size > 0
    img = mpimg.imread(path)
    assert img.size > 0
    assert np.isfinite(img).all()


@pytest.mark.skip(reason="Per-panel array vmin/vmax support lives on a different branch")
def test_plot_result_accepts_numpy_array_bounds(tmp_path):
    arr = np.arange(2 * 8 * 8, dtype=float).reshape(2, 8, 8) + 1.0
    out = tmp_path / "numpy_bounds.png"

    ju.plot_result(
        arr,
        output_file=str(out),
        colorbar=False,
        vmin=np.array([0.0, 1.0]),
        vmax=np.array([10.0, 11.0]),
    )

    _assert_written_image(out)


@pytest.mark.skip(reason="Per-panel array vmin/vmax support lives on a different branch")
def test_plot_result_accepts_jax_array_bounds(tmp_path):
    arr = np.arange(2 * 8 * 8, dtype=float).reshape(2, 8, 8) + 1.0
    out = tmp_path / "jax_bounds.png"

    ju.plot_result(
        arr,
        output_file=str(out),
        colorbar=False,
        vmin=jnp.array([0.0, 1.0]),
        vmax=jnp.array([10.0, 11.0]),
    )

    _assert_written_image(out)


def test_plot_result_accepts_scalar_bounds(tmp_path):
    arr = np.arange(2 * 8 * 8, dtype=float).reshape(2, 8, 8) + 1.0
    out = tmp_path / "scalar_bounds.png"

    ju.plot_result(
        arr,
        output_file=str(out),
        colorbar=False,
        vmin=0.0,
        vmax=20.0,
    )

    _assert_written_image(out)


def test_plot_result_accepts_2d_input(tmp_path):
    arr = np.arange(8 * 8, dtype=float).reshape(8, 8) + 1.0
    out = tmp_path / "single_image.png"

    ju.plot_result(arr, output_file=str(out), colorbar=False)

    _assert_written_image(out)


def test_plot_result_common_colorbar_writes_file(tmp_path):
    arr = np.stack(
        [
            np.arange(64, dtype=float).reshape(8, 8) + 1.0,
            np.arange(64, dtype=float).reshape(8, 8) + 10.0,
        ]
    )
    out = tmp_path / "common_colorbar.png"

    ju.plot_result(
        arr,
        output_file=str(out),
        colorbar=True,
        common_colorbar=True,
        vmin=0.0,  # ignored by common_colorbar branch
        vmax=1.0,
    )

    _assert_written_image(out)


def test_color_limits_are_shared_with_plot_result(monkeypatch, tmp_path):
    values = np.array([np.nan, -1.0, 0.0, 2.0, 8.0])
    assert _get_color_limits(values, logscale=True) == (2.0, 8.0)

    captured = []
    original = _get_color_limits

    def record_limits(*args, **kwargs):
        captured.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr("jubik.plot._get_color_limits", record_limits)
    ju.plot_result(
        values[1:].reshape(2, 2),
        output_file=str(tmp_path / "logscale.png"),
        colorbar=False,
        logscale=True,
    )
    assert captured


@pytest.mark.parametrize("nside", [1, 2, 8])
def test_get_nside_from_npix(nside):
    assert _get_nside_from_npix(12 * nside**2) == nside


def test_get_nside_from_invalid_npix_raises():
    with pytest.raises(ValueError, match="does not look like a HEALPix map"):
        _get_nside_from_npix(13)


def test_plot_healpix_result_uses_shared_limits_and_dark_style():
    first = np.arange(1, 49, dtype=float)
    maps = np.stack((first, 10.0 * first))

    fig, axes = ju.plot_healpix_result(
        maps,
        n_cols=2,
        common_colorbar=True,
        logscale=True,
        dark_background=True,
        xsize=64,
    )

    try:
        assert axes.shape == (1, 2)
        assert len(fig.axes) == 4
        assert all(isinstance(ax.images[0].norm, LogNorm) for ax in axes.flat)
        limits = [(ax.images[0].norm.vmin, ax.images[0].norm.vmax) for ax in axes.flat]
        assert limits[0] == limits[1]
        assert np.allclose(fig.get_facecolor()[:3], 0.0)
    finally:
        plt.close(fig)


def test_plot_healpix_result_validates_layout_and_flip():
    maps = np.ones((2, 12))

    with pytest.raises(ValueError, match="Layout has 1 panels"):
        ju.plot_healpix_result(maps, xsize=32)

    with pytest.raises(ValueError, match="Unsupported flip"):
        ju.plot_healpix_result(maps[0], flip="sideways", xsize=32)


@pytest.mark.parametrize("shape", [(64,), (2, 3, 4, 5)])
def test_plot_result_invalid_shape_raises(shape):
    arr = np.zeros(shape, dtype=float)

    with pytest.raises(ValueError, match="Wrong input shape"):
        ju.plot_result(arr, colorbar=False)


def test_plot_histograms_writes_file(tmp_path):
    hist = np.array([2.0, 1.0, 3.0])
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    out = tmp_path / "hist.png"

    ju.plot_histograms(hist, edges, filename=str(out), logx=False, logy=False)

    _assert_written_image(out)


def test_get_n_rows_from_n_samples_helper():
    assert _get_n_rows_from_n_samples(1) == 1
    assert _get_n_rows_from_n_samples(2) == 1
    assert _get_n_rows_from_n_samples(3) == 2
    assert _get_n_rows_from_n_samples(8) == 2
    assert _get_n_rows_from_n_samples(10) == 3
    assert _get_n_rows_from_n_samples(11) == 3
