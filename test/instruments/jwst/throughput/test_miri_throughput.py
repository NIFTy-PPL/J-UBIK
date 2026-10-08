"""MIRI: the half-power tophat fallback."""

import logging
from itertools import pairwise

import nifty.re as jft
import numpy as np
import pytest
from astropy import units as u

from jubik.color import Color
from jubik.instruments.jwst.data.jwst_information import JWST_FILTERS
from jubik.instruments.jwst.throughput.miri import miri_filter_weights


def _edges(edges) -> Color:
    edges = np.asarray(edges, float)
    return Color(np.c_[edges[:-1], edges[1:]] * u.um)


def test_tophat_over_the_half_power_range(caplog):
    _, _, _, blue, red = JWST_FILTERS["F560W"]
    edges = np.linspace(blue, red, 4)
    # nifty's logger does not propagate, so caplog listens on it directly
    jft.logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=jft.logger.name):
            band = miri_filter_weights(_edges(edges), "f560w")
    finally:
        jft.logger.removeHandler(caplog.handler)
    assert "F560W" in caplog.text and "half-power tophat" in caplog.text
    assert band.channels == slice(0, 3)
    expected = np.array([np.log(b / a) for a, b in pairwise(edges)])
    np.testing.assert_allclose(band.weights, expected / expected.sum(), rtol=1e-6)


def test_non_miri_filter_raises():
    with pytest.raises(KeyError, match="F444W"):
        miri_filter_weights(_edges([3.8, 5.0]), "F444W")
