"""Physical checks for the fixed Observation and radio-response conventions."""

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u
from astropy.constants import c
from astropy.coordinates import SkyCoord
from numpy.testing import assert_allclose

import jubik as ju
import jubik.instruments.resolve as rve
from jubik.instruments.resolve.data import AntennaPositions, Observation
from jubik.polarization import PolarizationType


def observation():
    return Observation(
        AntennaPositions(np.array([[3.0, 4.0, 0.0], [-6.0, 0.0, 8.0]])),
        vis=np.ones((1, 2, 2), dtype=np.complex128),
        weight=np.ones((1, 2, 2)),
        polarization=PolarizationType.I.get_legacy_polarization(),
        freq=np.array([1.0e9, 2.0e9]),
    )


def grid(angle_unit, frequency_unit):
    return ju.Grid(
        spatial=ju.wcs.WcsAstropy(
            center=SkyCoord(0.0 * u.deg, 0.0 * u.deg),
            shape=(16, 24),
            fov=u.Quantity([1.0, 1.5], u.deg).to(angle_unit),
        ),
        spectral=ju.Color(u.Quantity([0.9, 1.5, 2.1], u.GHz).to(frequency_unit)),
        polarization=PolarizationType.I,
    )


def test_effective_uvw_uses_metres_and_hertz():
    obs = observation()
    expected = obs.uvw.T[:, :, None] * obs.freq[None, None, :] / c.to_value(u.m / u.s)
    assert_allclose(obs.effective_uvw(), expected, rtol=1e-14)
    expected_lengths = np.linalg.norm(expected, axis=0)
    assert_allclose(obs.effective_uvwlen()[0], expected_lengths, rtol=1e-14)


@pytest.mark.parametrize("backend", ["ducc0", "finufft"])
@pytest.mark.parametrize(
    "angle_unit, frequency_unit",
    [
        pytest.param(u.rad, u.Hz, id="spatial-only"),
        pytest.param(u.deg, u.GHz, id="spectral-only"),
        pytest.param(u.rad, u.GHz, id="spatial-and-spectral"),
    ],
)
def test_response_is_independent_of_grid_unit_representation(
    backend, angle_unit, frequency_unit
):
    settings = (
        rve.parse.Ducc0Settings(epsilon=1e-9, do_wgridding=False, nthreads=1, verbosity=False)
        if backend == "ducc0"
        else rve.parse.FinufftSettings(epsilon=1e-9)
    )
    reference_grid = grid(u.deg, u.Hz)
    equivalent_grid = grid(angle_unit, frequency_unit)
    sky = np.random.default_rng(42).normal(size=reference_grid.shape)
    obs = observation()
    reference = rve.interferometry_response(obs, reference_grid, settings)(sky)
    equivalent = rve.interferometry_response(obs, equivalent_grid, settings)(sky)
    assert_allclose(equivalent, reference, rtol=1e-9, atol=1e-14)


def test_uvw_density_converts_pixel_scales_to_radians():
    dirty = import_module("jubik.instruments.resolve.dirty_image")
    # Supply only the spatial interface used by uvw_density. In radians the
    # pixel scales are 1/4 and 1/2, giving Fourier spacings of 1 and 1/3.
    sky_grid = SimpleNamespace(
        spatial=SimpleNamespace(
            shape_xy=(4, 6),
            pixel_scales_xy=u.Quantity([0.25, 0.5], u.rad).to(u.deg),
        )
    )
    eff_u = np.array([[-1.8, -1.2, 0.4, -0.3]])
    eff_v = np.array([[0.1, 0.2, -0.8, -0.5]])
    weights = np.array([[1.0, 2.0, 4.0, 8.0]])
    histogram, xedges, yedges = dirty.uvw_density(eff_u, eff_v, sky_grid, weights)

    assert_allclose(xedges, [-2.0, -1.0, 0.0, 1.0])
    assert_allclose(yedges, [-1.0, -2 / 3, -1 / 3, 0.0, 1 / 3, 2 / 3])
    # First two samples share a bin; the others occupy distinct bins. Unequal
    # axes, asymmetric coordinates and distinct weights expose swaps/sign flips.
    expected = np.array([
        [0.0, 0.0, 0.0, 3.0, 0.0],
        [0.0, 8.0, 0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, 0.0, 0.0],
    ])
    assert_allclose(histogram, expected)
    assert_allclose(histogram.sum(), weights.sum())
