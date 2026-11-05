import jax
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from numpy.testing import assert_allclose

import jubik as ju
import jubik.instruments.resolve as rve
from jubik.instruments.resolve.constants import SPEEDOFLIGHT

from generate_test_obs import generate_random_obs

jax.config.update("jax_enable_x64", True)

FREQ = 1e9
NPIX = 32
SETTINGS = rve.parse.FinufftSettings(epsilon=1e-12)


def grid(fov_y_deg=1.0):
    # Pixel 1 deg / 32 = 5.45e-4 rad, Nyquist 916 wavelengths = 275 m at 1 GHz.
    return ju.Grid(
        spatial=ju.wcs.WcsAstropy(
            center=SkyCoord(0 * u.deg, 0 * u.deg),
            shape=(NPIX, NPIX),
            fov=u.Quantity([1.0, fov_y_deg], u.deg),
        ),
        spectral=ju.color.Color.from_central_frequencies(np.array([FREQ])),
    )


def observation(freqs=(FREQ,), long_row=None, weight=1.0):
    """20 rows with |u|, |v| < 100 m, `long_row = (u, v)` replaces row 5 and
    `weight` (scalar or per channel) sets its weights."""
    obs = generate_random_obs(
        np.array(freqs),
        20,
        [-100, 100],
        [0, 0],
        ju.polarization.PolarizationType(("I",)),
        rng=np.random.default_rng(7),
    )
    uvw = obs.uvw.copy()
    weights = obs.weight_val.copy()
    if long_row is not None:
        uvw[5, :2] = long_row
        weights[:, 5] = weight
    return rve.Observation(
        rve.AntennaPositions(uvw),
        obs.vis_val,
        weights,
        obs.legacy_polarization,
        obs.freq,
        None,
    )


def test_a_baseline_beyond_nyquist_raises():
    with pytest.raises(ValueError, match="Nyquist frequency 916.7 wavelengths"):
        rve.interferometry_response(observation(long_row=(0, -400)), grid(), SETTINGS)


def test_a_flagged_baseline_beyond_nyquist_is_skipped():
    rve.interferometry_response(
        observation(long_row=(0, -400), weight=0.0), grid(), SETTINGS
    )


def test_the_highest_frequency_sets_the_limit():
    # 200 m is 667 wavelengths at 1 GHz and 1334 at 2 GHz, beyond 916.
    rve.interferometry_response(observation(long_row=(0, 200)), grid(), SETTINGS)
    two_freqs = ju.Grid(
        grid().spatial,
        ju.color.Color.from_central_frequencies(np.array([FREQ, 2 * FREQ])),
    )
    with pytest.raises(ValueError, match="Nyquist"):
        rve.interferometry_response(
            observation((FREQ, 2 * FREQ), long_row=(0, 200)), two_freqs, SETTINGS
        )


def test_a_flagged_highest_channel_does_not_set_the_limit():
    two_freqs = ju.Grid(
        grid().spatial,
        ju.color.Color.from_central_frequencies(np.array([FREQ, 2 * FREQ])),
    )
    rve.interferometry_response(
        observation((FREQ, 2 * FREQ), long_row=(0, 200), weight=[1.0, 0.0]),
        two_freqs,
        SETTINGS,
    )


def test_u_pairs_with_the_x_pixel_and_v_with_the_y_pixel():
    # Half the y fov halves dy: the Nyquist frequency is 916 in u, 1833 in v.
    narrow = grid(fov_y_deg=0.5)
    rve.interferometry_response(observation(long_row=(0, 400)), narrow, SETTINGS)
    with pytest.raises(ValueError, match=r"\|u\| baseline"):
        rve.interferometry_response(observation(long_row=(400, 0)), narrow, SETTINGS)


def test_within_nyquist_the_response_is_the_analytic_visibility():
    # A normalized gaussian of 2 pixels at the image centre has the
    # visibility exp(-2 pi^2 sigma^2 (u^2 + v^2)).
    obs = observation()
    g = grid()
    dx = g.spatial.pixel_scales_xy.to_value(u.rad)[0]
    sigma = 2 * dx
    pos = (np.arange(NPIX) - NPIX // 2) * dx
    image = np.exp(-(pos[:, None] ** 2 + pos[None, :] ** 2) / (2 * sigma**2))
    sky = np.zeros(g.shape) + image / (image.sum() * dx**2)

    response = rve.interferometry_response(obs, g, SETTINGS)
    uv = obs.uvw[:, :2] * FREQ / SPEEDOFLIGHT
    expected = np.exp(-2 * np.pi**2 * sigma**2 * (uv**2).sum(axis=1))
    assert_allclose(np.asarray(response(sky))[0, :, 0], expected, atol=1e-8)
