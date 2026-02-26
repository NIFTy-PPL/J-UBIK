import matplotlib
import matplotlib.image as mpimg
import numpy as np
import jax.numpy as jnp
import pytest

import jubik as ju
from jubik.plot import _get_n_rows_from_n_samples


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
