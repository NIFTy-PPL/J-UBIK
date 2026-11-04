import logging

import jax
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from numpy.testing import assert_allclose

import jubik as ju
import jubik.instruments.resolve as rve
from jubik.instruments.resolve.constants import SPEEDOFLIGHT
from jubik.instruments.resolve.likelihood.radio_likelihood import (
    restrict_to_nyquist,
)
from jubik.instruments.resolve.response import uv_beyond_nyquist
from nifty.cl.logger import logger

from generate_test_obs import generate_random_obs

jax.config.update("jax_enable_x64", True)

FREQ = 1e9
NPIX = 32


def grid():
    # Pixel 1 deg / 32 = 5.45e-4 rad, Nyquist 916 wavelengths = 275 m at 1 GHz.
    return ju.Grid(
        spatial=ju.wcs.WcsAstropy(
            center=SkyCoord(0 * u.deg, 0 * u.deg),
            shape=(NPIX, NPIX),
            fov=u.Quantity([1.0, 1.0], u.deg),
        ),
        spectral=ju.color.Color.from_central_frequencies(np.array([FREQ])),
    )


def observation_with_one_long_row():
    obs = generate_random_obs(
        np.array([FREQ]),
        20,
        [-200, 200],
        [0, 0],
        ju.polarization.PolarizationType(("I",)),
        rng=np.random.default_rng(7),
    )
    uvw = obs.uvw.copy()
    uvw[5, 1] = -400.0
    return rve.Observation(
        rve.AntennaPositions(uvw),
        obs.vis_val,
        obs.weight_val,
        obs.legacy_polarization,
        obs.freq,
        None,
    )


def test_mask_marks_the_long_row():
    mask = uv_beyond_nyquist(observation_with_one_long_row(), grid())
    assert np.flatnonzero(mask).tolist() == [5]


def test_raise_is_the_default_and_the_response_guards():
    obs = observation_with_one_long_row()
    with pytest.raises(ValueError, match="Nyquist"):
        restrict_to_nyquist(obs, grid(), "raise", "block")
    with pytest.raises(ValueError, match="Nyquist"):
        rve.interferometry_response(obs, grid(), rve.parse.FinufftSettings(1e-10))


def test_drop_removes_the_row_and_keeps_the_analytic_visibility(caplog):
    obs = observation_with_one_long_row()
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=logger.name):
            kept = restrict_to_nyquist(obs, grid(), "drop", "block")
    finally:
        logger.removeHandler(caplog.handler)
    assert kept.nrow == obs.nrow - 1
    assert "dropping 1 of 20 rows" in caplog.text

    # A normalized gaussian of 2 pixels at the image centre has the
    # visibility exp(-2 pi^2 sigma^2 (u^2 + v^2)).
    g = grid()
    dx = g.spatial.pixel_scales_xy.to_value(u.rad)[0]
    sigma = 2 * dx
    pos = (np.arange(NPIX) - NPIX // 2) * dx
    image = np.exp(-(pos[:, None] ** 2 + pos[None, :] ** 2) / (2 * sigma**2))
    sky = np.zeros(g.shape) + image / (image.sum() * dx**2)

    response = rve.interferometry_response(
        kept, g, rve.parse.FinufftSettings(epsilon=1e-12)
    )
    uv = kept.uvw[:, :2] * FREQ / SPEEDOFLIGHT
    expected = np.exp(-2 * np.pi**2 * sigma**2 * (uv**2).sum(axis=1))
    assert_allclose(np.asarray(response(sky))[0, :, 0], expected, atol=1e-8)


def test_unknown_policy_raises():
    with pytest.raises(ValueError, match="beyond_nyquist"):
        restrict_to_nyquist(observation_with_one_long_row(), grid(), "fold", "b")
