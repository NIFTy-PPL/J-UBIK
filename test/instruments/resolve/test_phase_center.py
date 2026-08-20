import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from numpy.testing import assert_allclose, assert_array_equal

import jubik as ju
import jubik.instruments.resolve as rve
from jubik.instruments.resolve.data.data_modify.phase_center import shift_phase_center
from jubik.instruments.resolve.parse.data.data_modify.phase_center import (
    ShiftObservation,
)

from generate_test_obs import generate_random_obs

pmp = pytest.mark.parametrize

SPEEDOFLIGHT = 299792458.0


def build_obs(n_rows=5, n_freq=3, uv_range=(-1.0e3, 1.0e3), w_range=(-5.0, 5.0)):
    """Random observation with `n_freq` channels of 0.1 GHz spacing."""
    pol_type = ju.polarization.PolarizationType(("LL", "RR"))
    freqs = 1.0e9 + np.arange(n_freq) * 0.1e9
    return generate_random_obs(freqs, n_rows, list(uv_range), list(w_range), pol_type)


def uv_in_wavelengths(obs):
    """(u, v) of every (row, channel) in wavelengths, shape (nrow, nfreq)."""
    uu = obs.uvw[:, 0, None] * obs.freq[None] / SPEEDOFLIGHT
    vv = obs.uvw[:, 1, None] * obs.freq[None] / SPEEDOFLIGHT
    return uu, vv


def replace_data(obs, vis, weight=None):
    """New Observation with the same antenna positions but different data."""
    if weight is None:
        weight = np.ones(vis.shape, dtype=np.float64)
    return rve.Observation(
        obs.antenna_positions,
        vis,
        weight,
        obs.legacy_polarization,
        obs.freq,
        None,
    )


# ---------------------------------------------------------------------------
# Round trip / no-op behaviour
# ---------------------------------------------------------------------------


@pmp("n_rows,n_freq", ((1, 1), (3, 1), (1, 4), (5, 3)))
def test_shift_then_unshift_is_identity(n_rows, n_freq):
    np.random.seed(100 + 10 * n_rows + n_freq)
    obs = build_obs(n_rows, n_freq)
    shift = [1.2e-4, -3.4e-4] * u.rad

    forward = shift_phase_center(obs, ShiftObservation(shift))
    back = shift_phase_center(forward, ShiftObservation(-shift))

    assert_allclose(back.vis_val, obs.vis_val, rtol=1e-12, atol=1e-12)
    # A phase rotation must not touch the amplitudes ...
    assert_allclose(np.abs(forward.vis_val), np.abs(obs.vis_val), rtol=1e-13)
    # ... nor the weights, nor the antenna positions.
    assert_array_equal(forward.weight_val, obs.weight_val)
    assert_array_equal(back.weight_val, obs.weight_val)
    assert forward.antenna_positions is obs.antenna_positions
    assert back.antenna_positions is obs.antenna_positions
    assert_array_equal(forward.freq, obs.freq)


def test_shift_none_is_noop():
    np.random.seed(101)
    obs = build_obs()

    assert shift_phase_center(obs, None) is obs


def test_zero_shift_leaves_visibilities_bitwise_identical():
    np.random.seed(102)
    obs = build_obs()

    new = shift_phase_center(obs, ShiftObservation([0.0, 0.0] * u.rad))

    assert_array_equal(new.vis_val, obs.vis_val)
    assert_array_equal(new.weight_val, obs.weight_val)


def test_shift_is_unit_aware():
    np.random.seed(103)
    obs = build_obs()
    shift_arcsec = [30.0, -12.5] * u.arcsec

    from_arcsec = shift_phase_center(obs, ShiftObservation(shift_arcsec))
    from_rad = shift_phase_center(obs, ShiftObservation(shift_arcsec.to(u.rad)))

    assert_allclose(from_arcsec.vis_val, from_rad.vis_val, rtol=1e-14, atol=1e-14)
    # A 30 arcsec shift is not the same as a 30 rad shift, i.e. the unit is
    # really taken into account.
    naive = shift_phase_center(obs, ShiftObservation(shift_arcsec.value * u.rad))
    assert not np.allclose(from_arcsec.vis_val, naive.vis_val)


# ---------------------------------------------------------------------------
# Sign / axis convention
# ---------------------------------------------------------------------------


def test_shift_prefactor_follows_the_imaging_convention():
    """The prefactor is exp(+2j pi (u * sx - v * sy)).

    The relative minus between the u and the v term is dictated by the
    imaging kernel (`flip_v=True` in `response.py`), see
    `test_shift_centers_point_source_of_the_jubik_response`.
    """
    np.random.seed(104)
    obs = build_obs(n_rows=4, n_freq=3)
    sx, sy = 3.0e-4, -7.0e-4

    new = shift_phase_center(obs, ShiftObservation([sx, sy] * u.rad))

    uu, vv = uv_in_wavelengths(obs)
    expected = obs.vis_val * np.exp(2j * np.pi * (uu * sx - vv * sy))[None]

    assert_allclose(new.vis_val, expected, rtol=1e-12, atol=1e-12)


def build_response_setup(npix=64, fov_deg=1.0, freqs=(1.0e9, 1.5e9), n_rows=6):
    """Grid + w == 0 observation + ducc/finufft responses of jubik's resolve."""
    freqs = np.array(freqs)
    pol_type = ju.polarization.PolarizationType(("LL", "RR"))
    obs = generate_random_obs(freqs, n_rows, [-3.0e3, 3.0e3], [0.0, 0.0], pol_type)

    # A NaN sky center makes `calculate_phase_offset_to_image_center` return
    # (0, 0), i.e. the response does not apply a phase shift of its own.
    sky_center = SkyCoord(ra=np.nan * u.rad, dec=np.nan * u.rad)
    fov = u.Quantity((fov_deg * u.deg, fov_deg * u.deg))
    spatial = ju.wcs.WcsAstropy(center=sky_center, shape=(npix, npix), fov=fov)
    spectral = ju.color.Color.from_central_frequencies(freqs)
    grid = ju.Grid(
        spatial=spatial,
        spectral=spectral,
        polarization=ju.polarization.PolarizationType(("I",)),
    )

    ducc_settings = rve.parse.Ducc0Settings(
        epsilon=1e-11, do_wgridding=False, nthreads=1, verbosity=False
    )
    finufft_settings = rve.parse.FinufftSettings(epsilon=1e-11)
    r_ducc = rve.interferometry_response(obs, grid, backend_settings=ducc_settings)
    r_finufft = rve.interferometry_response(
        obs, grid, backend_settings=finufft_settings
    )
    return grid, obs, r_ducc, r_finufft


def point_source_sky(grid, di, dj):
    """Unit flux delta `di`/`dj` pixels off the image center."""
    sky = np.zeros(grid.shape)
    cx, cy = grid.shape[3] // 2, grid.shape[4] // 2
    dvol = grid.spatial.dvol.to(u.rad**2).value
    sky[:, :, :, cx + di, cy + dj] = 1.0 / dvol
    return sky


@pmp("di,dj", ((3, 5), (-4, 2)))
def test_response_phase_convention_is_exp_minus_2jpi_ul_minus_vm(di, dj):
    """Pin down the convention of jubik's own instrument response."""
    np.random.seed(105 + di + dj)
    grid, obs, r_ducc, r_finufft = build_response_setup()

    vis_ducc = np.asarray(r_ducc(point_source_sky(grid, di, dj)))
    vis_finufft = np.asarray(r_finufft(point_source_sky(grid, di, dj)))

    dx, dy = grid.spatial.distances.to(u.rad).value
    ll, mm = di * dx, dj * dy
    uu, vv = uv_in_wavelengths(obs)
    expected = np.broadcast_to(
        np.exp(-2j * np.pi * (uu * ll - vv * mm))[None], vis_ducc.shape
    )

    assert_allclose(vis_ducc, expected, rtol=0, atol=1e-9)
    assert_allclose(vis_finufft, expected, rtol=0, atol=1e-9)
    # The wrong relative sign between u and v is a completely different data
    # set, i.e. this test really constrains the convention.
    wrong = np.exp(-2j * np.pi * (uu * ll + vv * mm))[None]
    assert np.max(np.abs(vis_ducc - wrong)) > 1.0


@pmp("di,dj", ((3, 5), (-4, 2)))
def test_shift_centers_point_source_of_the_jubik_response(di, dj):
    """Shifting by the source offset must center the source.

    A source sitting in the image center produces vis == 1 everywhere, so
    after the shift the visibilities must be exactly that.
    """
    np.random.seed(106 + di + dj)
    grid, obs, r_ducc, _ = build_response_setup()

    dx, dy = grid.spatial.distances.to(u.rad).value
    ll, mm = di * dx, dj * dy

    data = replace_data(obs, np.asarray(r_ducc(point_source_sky(grid, di, dj))))
    centered = shift_phase_center(data, ShiftObservation([ll, mm] * u.rad))

    vis_center = np.asarray(r_ducc(point_source_sky(grid, 0, 0)))
    assert_allclose(vis_center, np.ones(vis_center.shape), rtol=0, atol=1e-9)
    assert_allclose(centered.vis_val, vis_center, rtol=0, atol=1e-8)


def test_shift_moves_center_source_to_the_given_offset():
    """The inverse statement: shifting by -(l, m) moves a centered source out."""
    np.random.seed(107)
    di, dj = 2, -6
    grid, obs, r_ducc, _ = build_response_setup()

    dx, dy = grid.spatial.distances.to(u.rad).value
    ll, mm = di * dx, dj * dy

    data = replace_data(obs, np.asarray(r_ducc(point_source_sky(grid, 0, 0))))
    moved = shift_phase_center(data, ShiftObservation([-ll, -mm] * u.rad))

    expected = np.asarray(r_ducc(point_source_sky(grid, di, dj)))
    assert_allclose(moved.vis_val, expected, rtol=0, atol=1e-8)


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------


def test_shift_preserves_single_precision():
    np.random.seed(108)
    obs = build_obs(n_rows=4, n_freq=1)
    single = rve.Observation(
        obs.antenna_positions,
        obs.vis_val.astype(np.complex64),
        obs.weight_val.astype(np.float32),
        obs.legacy_polarization,
        obs.freq,
        None,
    )
    shift = ShiftObservation([1.0e-4, 2.0e-4] * u.rad)

    new = shift_phase_center(single, shift)

    assert new.vis_val.dtype == np.complex64
    assert new.weight_val.dtype == np.float32
    double = shift_phase_center(obs, shift)
    assert_allclose(new.vis_val, double.vis_val, rtol=1e-5, atol=1e-6)


def test_shift_preserves_double_precision():
    np.random.seed(109)
    obs = build_obs(n_rows=4, n_freq=2)

    new = shift_phase_center(obs, ShiftObservation([1.0e-4, 2.0e-4] * u.rad))

    assert new.vis_val.dtype == np.complex128
    assert new.weight_val.dtype == np.float64
