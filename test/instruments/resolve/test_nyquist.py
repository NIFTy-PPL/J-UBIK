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


def two_freqs():
    return ju.Grid(
        grid().spatial,
        ju.color.Color.from_central_frequencies(np.array([FREQ, 2 * FREQ])),
    )


def test_the_highest_frequency_sets_the_limit():
    # 200 m is 667 wavelengths at 1 GHz and 1334 at 2 GHz, beyond 916.
    rve.interferometry_response(observation(long_row=(0, 200)), grid(), SETTINGS)
    with pytest.raises(ValueError, match="Nyquist"):
        rve.interferometry_response(
            observation((FREQ, 2 * FREQ), long_row=(0, 200)), two_freqs(), SETTINGS
        )


def test_a_flagged_highest_channel_does_not_set_the_limit():
    rve.interferometry_response(
        observation((FREQ, 2 * FREQ), long_row=(0, 200), weight=[1.0, 0.0]),
        two_freqs(),
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


def modify_of(cfg):
    """The `ObservationModify` of the one file of a data block `cfg`."""
    data_cfg = {"data_path": "/data", "data_templates": ["a.npz"]} | cfg
    (file,) = DataLoading.from_yaml_dict(data_cfg).data_files
    return build_observation_modify(
        ObservationModifyConfig.from_yaml_dict(data_cfg), file
    )


def test_raise_is_the_default():
    obs = observation(long_row=(0, -400))
    assert ObservationModifyConfig.from_yaml_dict({}).beyond_nyquist == BeyondNyquist()
    with pytest.raises(ValueError, match="Nyquist"):
        modify_observation(grid(), obs, modify_of({}))


def test_unknown_values_fail_at_parse():
    assert BeyondNyquist.from_yaml_dict("drop") == BeyondNyquist(drop=True)
    with pytest.raises(ValueError, match="beyond_nyquist"):
        ObservationModifyConfig.from_yaml_dict({"beyond_nyquist": "fold"})


def test_drop_keeps_exactly_the_rows_the_guard_accepts(caplog):
    # At 1 GHz and 2 GHz, the Nyquist frequency is 916 wavelengths. Rows:
    # 5 beyond (400 m), 6 beyond but flagged, 7 beyond only at the flagged
    # 2 GHz channel, 8 beyond at 2 GHz (200 m = 1334 wavelengths).
    obs = observation((FREQ, 2 * FREQ))
    uvw, weights = obs.uvw.copy(), obs.weight_val.copy()
    uvw[5:9, :2] = [(0, -400), (0, -400), (0, 200), (200, 0)]
    weights[:, 6] = 0.0
    weights[:, 7, 1] = 0.0
    obs = rve.Observation(
        rve.AntennaPositions(uvw),
        obs.vis_val,
        weights,
        obs.legacy_polarization,
        obs.freq,
        None,
    )
    assert np.flatnonzero(uv_beyond_nyquist(obs, two_freqs())).tolist() == [5, 8]

    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=logger.name):
            new = modify_observation(
                two_freqs(), obs, modify_of({"beyond_nyquist": "drop"})
            )
    finally:
        logger.removeHandler(caplog.handler)

    kept = [i for i in range(obs.nrow) if i not in (5, 8)]
    assert_allclose(new.uvw, obs.uvw[kept])
    assert "Dropping 2 of 20 rows" in caplog.text
    assert "longest kept baseline is 0.4 km" in caplog.text
    rve.interferometry_response(new, two_freqs(), SETTINGS)


def test_a_row_too_long_only_at_a_channel_spectral_removes_is_kept():
    # 200 m is 667 wavelengths at 1 GHz and 1334 at 2 GHz; `spectral` removes
    # the 2 GHz channel before the check.
    obs = observation((FREQ, 2 * FREQ), long_row=(0, 200))
    restricted = {"spectral": {"min": 0.5 * FREQ, "max": 1.5 * FREQ}}

    with pytest.raises(ValueError, match="Nyquist"):
        modify_observation(two_freqs(), obs, modify_of({}))
    new = modify_observation(two_freqs(), obs, modify_of(restricted))

    assert new.nrow == obs.nrow and new.freq.tolist() == [FREQ]
    rve.interferometry_response(new, grid(), SETTINGS)
