"""Packaged NIRCam curves, the tophat fallback and the repacking helper."""

import logging

import nifty.re as jft
import numpy as np
import pytest
from astropy import units as u

from jubik.color import Color
from jubik.instruments.jwst.data import throughput
from jubik.instruments.jwst.data.jwst_information import JWST_FILTERS
from jubik.instruments.jwst.data.throughput import (
    THROUGHPUT_VERSION,
    jwst_transmission,
    pack_throughputs,
)
from jubik.sky_filter import BandWeights


def test_f444w_curve_is_packaged():
    t = jwst_transmission("f444w")
    assert THROUGHPUT_VERSION == "nircam_throughputs_4Nov2022_v5"
    assert t.throughput.max() > 0.3
    support = t.wavelength_um[t.throughput > 0]
    assert 3.6 <= support.min() and support.max() <= 5.2


def test_f444w_weights_on_four_channels():
    t = jwst_transmission("F444W")
    # the red wing past 5.0 um holds 1.5% of the passband, above MAX_MISSING
    with pytest.raises(ValueError, match="F444W"):
        BandWeights.from_band(Color(np.linspace(3.8, 5.0, 5) * u.um), t, name="F444W")
    band_weights = BandWeights.from_band(Color(np.linspace(3.7, 5.1, 5) * u.um), t)
    w = band_weights.weights[0]
    assert band_weights.channels == slice(0, 4) and band_weights.weights.shape == (1, 4)
    np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-12)
    assert min(w[1], w[2]) > max(w[0], w[3])


def test_miri_falls_back_to_tophat(caplog):
    # nifty's logger does not propagate, so caplog listens on it directly
    jft.logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=jft.logger.name):
            t = jwst_transmission("F560W")
    finally:
        jft.logger.removeHandler(caplog.handler)
    _, _, _, blue, red = JWST_FILTERS["F560W"]
    np.testing.assert_array_equal(t.wavelength_um, [blue, red])
    np.testing.assert_array_equal(t.throughput, [1.0, 1.0])
    assert "F560W" in caplog.text and "half-power tophat" in caplog.text


def test_unknown_filter_raises():
    with pytest.raises(KeyError, match="F999W"):
        jwst_transmission("F999W")


def test_pack_throughputs_round_trip(tmp_path):
    lam = np.linspace(1.0, 2.0, 5)
    T = np.linspace(0.1, 0.5, 5)
    np.savetxt(
        tmp_path / "f150w_mean_system_throughput.txt",
        np.c_[lam, T],
        header="Microns Throughput",
        comments="",
    )
    out = tmp_path / "curves.npz"
    assert pack_throughputs(tmp_path, "test_tag", out) == 1
    with np.load(out) as npz:
        assert str(npz["version"]) == "test_tag"
        np.testing.assert_allclose(npz["F150W_lam_um"], lam)
        np.testing.assert_allclose(npz["F150W_T"], T, rtol=1e-6)
    with pytest.raises(FileNotFoundError):
        pack_throughputs(tmp_path / "empty", "test_tag", out)


def test_unknown_module_attribute_raises():
    with pytest.raises(AttributeError):
        _ = throughput.NOT_A_NAME
