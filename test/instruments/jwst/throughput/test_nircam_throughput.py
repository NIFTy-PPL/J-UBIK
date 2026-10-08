"""NIRCam: the packaged STScI curves and the repacking helper."""

import numpy as np
import pytest
from astropy import units as u

from jubik.color import Color
from jubik.instruments.jwst.throughput import nircam
from jubik.instruments.jwst.throughput.nircam import (
    THROUGHPUT_VERSION,
    nircam_filter_weights,
    pack_throughputs,
)


def _edges(edges) -> Color:
    edges = np.asarray(edges, float)
    return Color(np.c_[edges[:-1], edges[1:]] * u.um)


def test_packaged_release():
    assert THROUGHPUT_VERSION == "nircam_throughputs_4Nov2022_v5"


def test_f444w_weights_on_four_channels():
    # the red wing past 5.0 um holds 1.5% of the passband, above MAX_MISSING
    with pytest.raises(ValueError, match="F444W"):
        nircam_filter_weights(_edges(np.linspace(3.8, 5.0, 5)), "f444w")
    band = nircam_filter_weights(_edges(np.linspace(3.7, 5.1, 5)), "F444W")
    w = band.weights
    assert band.channels == slice(0, 4) and w.shape == (4,)
    np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-12)
    assert min(w[1], w[2]) > max(w[0], w[3])


def test_non_nircam_filter_raises():
    with pytest.raises(KeyError, match="F560W"):
        nircam_filter_weights(_edges([5.0, 6.2]), "F560W")


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
        _ = nircam.NOT_A_NAME
