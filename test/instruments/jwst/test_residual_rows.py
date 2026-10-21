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
