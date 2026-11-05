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
from jubik.instruments.resolve.data.data_modify.flagging import uv_beyond_nyquist
from jubik.instruments.resolve.data.data_modify.pipeline import (
    build_observation_modify,
    modify_observation,
)
from jubik.instruments.resolve.parse.data import DataLoading
from jubik.instruments.resolve.parse.data.data_modify.flagging import (
    BeyondNyquist,
)
from jubik.instruments.resolve.parse.data.data_modify.pipeline import (
    ObservationModifyConfig,
)
from nifty.re import logger

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


def modify_of(cfg):
    """The `ObservationModify` of the one file of a data block `cfg`."""
    data_cfg = {"data_path": "/data", "data_templates": ["a.npz"]} | cfg
    (file,) = DataLoading.from_yaml_dict(data_cfg).data_files
    return build_observation_modify(
        ObservationModifyConfig.from_yaml_dict(data_cfg), file
    )


def test_mask_marks_the_long_row():
    mask = uv_beyond_nyquist(observation_with_one_long_row(), grid())
    assert np.flatnonzero(mask).tolist() == [5]


def test_the_pipeline_raises_by_default_and_the_response_guards():
    obs = observation_with_one_long_row()
    with pytest.raises(ValueError, match="Nyquist"):
        modify_observation(grid(), obs, modify_of({}))
    with pytest.raises(ValueError, match="Nyquist"):
        rve.interferometry_response(obs, grid(), rve.parse.FinufftSettings(1e-10))


def test_the_pipeline_drops_the_row_and_keeps_the_analytic_visibility(caplog):
    obs = observation_with_one_long_row()
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=logger.name):
            kept = modify_observation(
                grid(), obs, modify_of({"beyond_nyquist": "drop"})
            )
    finally:
        logger.removeHandler(caplog.handler)
    assert kept.nrow == obs.nrow - 1
    assert "Dropping 1 of 20 rows" in caplog.text

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


def test_beyond_nyquist_parsing():
    assert BeyondNyquist.from_yaml_dict(None) == BeyondNyquist(drop=False)
    assert BeyondNyquist.from_yaml_dict("drop") == BeyondNyquist(drop=True)
    with pytest.raises(ValueError, match="beyond_nyquist"):
        BeyondNyquist.from_yaml_dict("fold")


def test_the_pipeline_checks_after_the_frequency_restriction():
    # The long row is beyond Nyquist only at 2 GHz: 200 m is 667 wavelengths
    # at 1 GHz, 1334 at 2 GHz, the Nyquist frequency is 916.
    obs = generate_random_obs(
        np.array([FREQ, 2 * FREQ]),
        20,
        [-100, 100],
        [0, 0],
        ju.polarization.PolarizationType(("I",)),
        rng=np.random.default_rng(7),
    )
    uvw = obs.uvw.copy()
    uvw[5, 1] = -200.0
    obs = rve.Observation(
        rve.AntennaPositions(uvw),
        obs.vis_val,
        obs.weight_val,
        obs.legacy_polarization,
        obs.freq,
        None,
    )
    with pytest.raises(ValueError, match="Nyquist"):
        modify_observation(grid(), obs, modify_of({}))
    kept = modify_observation(
        grid(), obs, modify_of({"spectral": {"min": 0.5 * FREQ, "max": 1.5 * FREQ}})
    )
    assert kept.nrow == obs.nrow and kept.freq.tolist() == [FREQ]
