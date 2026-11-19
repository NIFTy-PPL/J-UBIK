"""Residual plot rows follow the band order, not the sky channel index."""

import numpy as np
from astropy import units as u

from jubik.grid import Grid
from jubik.instruments.jwst.plotting.residuals import _determine_ypos
from jubik.instruments.jwst.throughput.weights import Throughput, throughput_weights
from jubik.sky_filter import SkyFilter


def test_rows_enumerate_bands():
    # 100 channels over 0.5 to 5.5 um; F277W-like and F444W-like curves start
    # deep inside the grid and far apart, which once gave rows 0 and 27
    edges = np.linspace(0.5, 5.5, 101)
    grid = Grid.from_shape_and_fov(
        shape=(4, 4), fov=(1 * u.arcsec, 1 * u.arcsec), frequencies=edges * u.um
    )
    bands = {
        "f277w": throughput_weights(
            grid.spectral, Throughput([2.4, 3.1] * u.um, np.ones(2))
        ),
        "f444w": throughput_weights(
            grid.spectral, Throughput([3.9, 5.0] * u.um, np.ones(2))
        ),
    }
    sky_filter = SkyFilter(grid, bands)
    assert bands["f444w"].channels.start > 1
    assert [_determine_ypos(b, sky_filter) for b in bands] == [0, 1]


def test_residual_limits_symmetric_about_zero():
    # no limit, no norm: zero is the colour map's centre, whatever the residual's range
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from jubik.instruments.jwst.parse.plotting import FieldPlottingConfig, ResidualPlottingConfig
    from jubik.instruments.jwst.plotting.plotting_base import plot_data_data_model_residuals

    cfg = ResidualPlottingConfig.from_yaml_dict(
        dict(residual=dict(vmin=None, vmax=None, norm=None, cmap="RdBu_r"))
    )
    data = np.array([[-12.0, 0.0], [6.0, np.nan]])  # nan: a masked pixel
    _, axes = plt.subplots(1, 3)
    ims = plot_data_data_model_residuals(
        [None] * 3, axes, "f115w", data, np.zeros_like(data), 1.0, True, cfg.residual, FieldPlottingConfig()
    )
    assert ims[2].get_clim() == (-12.0, 12.0)
    plt.close("all")
    # a set limit is kept, and so is the default of -3 to 3
    assert FieldPlottingConfig(vmin=-5).residual_kwargs()["norm"] is None
    assert ResidualPlottingConfig.from_yaml_dict({}).residual.residual_kwargs()["vmax"] == 3
