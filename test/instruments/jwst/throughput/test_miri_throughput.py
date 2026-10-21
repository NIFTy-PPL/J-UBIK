"""MIRI: the half-power tophat fallback."""

import logging

import nifty.re as jft
import numpy as np
import pytest
from astropy import units as u

from jubik.instruments.jwst.data.jwst_information import JWST_FILTERS
from jubik.instruments.jwst.throughput.miri import miri_throughput


def test_tophat_over_the_half_power_range(caplog):
    _, _, _, blue, red = JWST_FILTERS["F560W"]
    # nifty's logger does not propagate, so caplog listens on it directly
    jft.logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=jft.logger.name):
            curve = miri_throughput("f560w")
    finally:
        jft.logger.removeHandler(caplog.handler)
    assert "F560W" in caplog.text and "half-power tophat" in caplog.text
    np.testing.assert_allclose(curve.wavelength.to_value(u.um), [blue, red])
    np.testing.assert_allclose(curve.throughput, [1.0, 1.0])


def test_non_miri_filter_raises():
    with pytest.raises(KeyError, match="F444W"):
        miri_throughput("F444W")
