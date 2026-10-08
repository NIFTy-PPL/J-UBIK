"""jwst_filter_weights: dispatch on the instrument tables."""

import numpy as np
import pytest
from astropy import units as u

from jubik.color import Color
from jubik.instruments.jwst.throughput import jwst_filter_weights


def _edges(edges) -> Color:
    edges = np.asarray(edges, float)
    return Color(np.c_[edges[:-1], edges[1:]] * u.um)


def _write_f444w(directory):
    directory.mkdir(exist_ok=True)
    wavelength_um = np.linspace(3.7, 5.1, 141)
    throughput = 0.5 * np.sin(np.pi * (wavelength_um - 3.7) / 1.4) ** 2
    np.savetxt(
        directory / "F444W_mean_system_throughput.txt",
        np.c_[wavelength_um, throughput],
        header="Microns Throughput",
        comments="",
    )


def test_nircam_filter_reads_the_throughput_dir(tmp_path):
    _write_f444w(tmp_path)
    band = jwst_filter_weights(_edges(np.linspace(3.7, 5.1, 5)), "f444w", tmp_path)
    assert band.channels == slice(0, 4)
    assert not np.allclose(band.weights, 0.25)  # a real curve, not a tophat


def test_nircam_filter_without_throughput_dir_raises():
    with pytest.raises(ValueError, match="throughput_dir.*--download"):
        jwst_filter_weights(_edges([3.7, 5.1]), "F444W")


def test_miri_filter_uses_the_tophat_and_ignores_the_dir(tmp_path):
    band = jwst_filter_weights(_edges([5.054, 5.6, 6.171]), "F560W", tmp_path)
    assert band.channels == slice(0, 2)
    expected = np.log([5.6 / 5.054, 6.171 / 5.6])
    np.testing.assert_allclose(band.weights, expected / expected.sum(), rtol=1e-6)


def test_unknown_filter_raises():
    with pytest.raises(KeyError, match="F999W"):
        jwst_filter_weights(_edges([1.0, 2.0]), "F999W")
