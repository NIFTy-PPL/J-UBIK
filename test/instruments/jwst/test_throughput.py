"""JWST filter weights: packaged NIRCam curves, the MIRI tophat fallback, the
photon-weighted channel integrals, and the repacking helper."""

import logging
from itertools import pairwise

import nifty.re as jft
import numpy as np
import pytest
from astropy import units as u

from jubik.color import Color
from jubik.instruments.jwst.data import throughput
from jubik.instruments.jwst.data.jwst_information import JWST_FILTERS
from jubik.instruments.jwst.data.throughput import (
    MAX_MISSING,
    THROUGHPUT_VERSION,
    jwst_filter_weights,
    pack_throughputs,
    throughput_weights,
)

# rising edge 3.9 to 4.1 um, flat to 5.0 um
RAMP = (np.array([3.9, 4.1, 5.0]), np.array([0.0, 1.0, 1.0]))


def _tophat(lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    return np.array([lo, hi]), np.ones(2)


def _spectral(bounds) -> Color:
    return Color(np.asarray(bounds, float) * u.um)


def _edges(edges) -> Color:
    edges = np.asarray(edges, float)
    return _spectral(np.c_[edges[:-1], edges[1:]])


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


def _log_ratio_weights(edges, lo, hi):
    """Normalised tophat weights ln(hi_i / lo_i) over the channels inside [lo, hi]."""
    w = np.array(
        [
            np.log(min(b, hi) / max(a, lo)) if min(b, hi) > max(a, lo) else 0.0
            for a, b in pairwise(edges)
        ]
    )
    return w / w.sum()


def _warnings(fn, caplog):
    # nifty's logger does not propagate, so caplog listens on it directly
    jft.logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=jft.logger.name):
            return fn()
    finally:
        jft.logger.removeHandler(caplog.handler)


# --------------------------------------------------------------------------- #
# jwst_filter_weights: NIRCam curves and the MIRI fallback
# --------------------------------------------------------------------------- #


def test_packaged_release():
    assert THROUGHPUT_VERSION == "nircam_throughputs_4Nov2022_v5"


def test_f444w_weights_on_four_channels():
    # the red wing past 5.0 um holds 1.5% of the passband, above MAX_MISSING
    with pytest.raises(ValueError, match="F444W"):
        jwst_filter_weights(_edges(np.linspace(3.8, 5.0, 5)), "f444w")
    band = jwst_filter_weights(_edges(np.linspace(3.7, 5.1, 5)), "F444W")
    w = band.weights
    assert band.channels == slice(0, 4) and w.shape == (4,)
    np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-12)
    assert min(w[1], w[2]) > max(w[0], w[3])


def test_miri_falls_back_to_tophat(caplog):
    _, _, _, blue, red = JWST_FILTERS["F560W"]
    edges = np.linspace(blue, red, 4)
    band = _warnings(lambda: jwst_filter_weights(_edges(edges), "f560w"), caplog)
    assert "F560W" in caplog.text and "half-power tophat" in caplog.text
    assert band.channels == slice(0, 3)
    np.testing.assert_allclose(
        band.weights, _log_ratio_weights(edges, blue, red), rtol=1e-6
    )


def test_unknown_filter_raises():
    with pytest.raises(KeyError, match="F999W"):
        jwst_filter_weights(_edges([1.0, 2.0]), "F999W")


def test_one_channel_per_filter_is_one_hot():
    names = ["F150W", "F277W", "F444W"]
    spectral = _spectral([JWST_FILTERS[n][3:5] for n in names])
    for index, n in enumerate(names):
        band = throughput_weights(spectral, *_tophat(*JWST_FILTERS[n][3:5]), name=n)
        assert band.channels == slice(index, index + 1)
        np.testing.assert_allclose(band.weights, [1.0])


# --------------------------------------------------------------------------- #
# throughput_weights: the photon-weighted integrals
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bounds",
    [
        [[3.5, 4.3], [4.3, 5.2]],  # case 1: edge cuts one channel boundary
        [[3.5, 4.0], [4.0, 4.6], [4.6, 5.2]],  # case 2: edge inside the first channel
    ],
)
def test_soft_edge_gives_partial_weights(bounds):
    band = throughput_weights(_spectral(bounds), *RAMP)
    expected = np.array([_ramp_integral(lo, hi) for lo, hi in bounds])
    expected /= expected.sum()
    assert band.channels == slice(0, len(bounds)) and band.weights.shape == (
        len(bounds),
    )
    np.testing.assert_allclose(band.weights, expected, rtol=1e-5)
    np.testing.assert_allclose(band.weights.sum(), 1.0, rtol=1e-12)


def test_tophat_weights_are_log_ratios():
    edges = np.array([1.0, 1.5, 2.0, 3.0])
    band = throughput_weights(_edges(edges), *_tophat(1.2, 2.4))
    np.testing.assert_allclose(
        band.weights, _log_ratio_weights(edges, 1.2, 2.4), rtol=1e-6
    )


def test_curve_inside_one_channel_is_one_hot():
    band = throughput_weights(_edges(np.linspace(1.0, 2.0, 11)), *_tophat(1.32, 1.38))
    assert band.channels == slice(3, 4)
    np.testing.assert_allclose(band.weights, [1.0])


def test_missing_passband_raises_above_max_missing():
    with pytest.raises(ValueError, match="biased.*outside the sky"):
        throughput_weights(_spectral([[4.0, 5.2]]), *RAMP, name="biased")
    with pytest.raises(ValueError, match="gap"):
        throughput_weights(_spectral([[3.5, 4.0], [4.3, 5.2]]), *RAMP, name="gap")


def test_small_missing_passband_warns(caplog):
    # overhang holding about half of MAX_MISSING
    hi = 2.0 * (1 + 0.5 * MAX_MISSING * np.log(2.0 / 1.5))
    spectral = _spectral([[1.0, 1.5], [1.5, 2.0]])
    band = _warnings(
        lambda: throughput_weights(spectral, *_tophat(1.5, hi), name="wing"), caplog
    )
    assert band.channels == slice(1, 2)
    np.testing.assert_allclose(band.weights, [1.0])
    assert "wing" in caplog.text and "renormalised" in caplog.text
    with pytest.raises(ValueError, match="wing"):
        throughput_weights(spectral, *_tophat(1.5, hi), max_missing=0.0, name="wing")


def test_full_coverage_does_not_warn(caplog):
    spectral = _spectral([[1.0, 1.5], [1.5, 2.0]])
    _warnings(
        lambda: throughput_weights(spectral, *_tophat(1.5, 2.0), max_missing=0.0),
        caplog,
    )
    inside = (np.array([1.2, 1.4, 1.9]), np.array([0.0, 1.0, 1.0]))
    _warnings(lambda: throughput_weights(spectral, *inside, max_missing=0.0), caplog)
    assert caplog.text == ""


def test_malformed_curves_raise():
    spectral = _edges([1.0, 2.0])
    with pytest.raises(ValueError, match="ascending"):
        throughput_weights(spectral, [1.0, 1.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="same length"):
        throughput_weights(spectral, [1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="non-negative"):
        throughput_weights(spectral, [1.0, 2.0], [0.0, 0.0])


# --------------------------------------------------------------------------- #
# pack_throughputs and the module attributes
# --------------------------------------------------------------------------- #


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
