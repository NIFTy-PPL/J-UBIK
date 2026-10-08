"""jwst_filter_weights: dispatch on the instrument tables."""

import numpy as np
import pytest
from astropy import units as u

from jubik.color import Color
from jubik.instruments.jwst.throughput import jwst_filter_weights


def _edges(edges) -> Color:
    edges = np.asarray(edges, float)
    return Color(np.c_[edges[:-1], edges[1:]] * u.um)


def test_nircam_filter_uses_the_packaged_curve():
    band = jwst_filter_weights(_edges(np.linspace(3.7, 5.1, 5)), "f444w")
    assert band.channels == slice(0, 4)
    assert not np.allclose(band.weights, 0.25)  # a real curve, not a tophat


def test_miri_filter_uses_the_tophat():
    band = jwst_filter_weights(_edges([5.054, 5.6, 6.171]), "F560W")
    assert band.channels == slice(0, 2)
    expected = np.log([5.6 / 5.054, 6.171 / 5.6])
    np.testing.assert_allclose(band.weights, expected / expected.sum(), rtol=1e-6)


def test_unknown_filter_raises():
    with pytest.raises(KeyError, match="F999W"):
        jwst_filter_weights(_edges([1.0, 2.0]), "F999W")
