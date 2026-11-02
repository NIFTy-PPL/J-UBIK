from copy import deepcopy

import jax
import nifty.re as jft
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from numpy.testing import assert_allclose

import jubik as ju
import jubik.instruments.resolve as rve
from jubik.instruments.resolve.calibration import build_gain
from jubik.instruments.resolve.data.auxiliary_table import AuxiliaryTable
from jubik.instruments.resolve.likelihood import build_radio_likelihood
from jubik.instruments.resolve.likelihood.mosaic_likelihood import (
    VariableLikelihoodBuilder,
    build_likelihood_from_sky_beamer,
)
from jubik.instruments.resolve.mosaicing.sky_beamer import BeamPattern, SkyBeamer
from jubik.instruments.resolve.parse.calibration import CalibrationConfig, TrackPrior
from jubik.instruments.resolve.parse.noise import BaseLineCorrection

from generate_test_obs import generate_random_obs

jax.config.update("jax_enable_x64", True)

FREQS = np.array([1e9, 1.3e9])
DT = 20.0
T0 = 5.0e9  # absolute measurement-set times
CALIBRATION = {
    "dt": DT,
    "phase": {"intercept": [0.0, 0.5], "slope": [0.0, 0.0], "deviations": [0.5, 0.2]},
    "log_amplitude": {
        "intercept": [0.0, 0.1],
        "slope": [0.0, 0.0],
        "deviations": [0.1, 0.05],
    },
}
CONFIG = CalibrationConfig.from_yaml_dict(CALIBRATION)
POL = ju.polarization.PolarizationType(("LL", "RR"))
FIELD = "band7_fld"


def gain_obs(ant1, ant2, steps):
    """Observation of one row per antenna pair at the times `T0 + steps * DT`."""
    return generate_random_obs(
        FREQS,
        len(ant1),
        [-1e2, 1e2],
        [-5, 5],
        POL,
        ant1=np.asarray(ant1),
        ant2=np.asarray(ant2),
        times=T0 + DT * np.asarray(steps, dtype=float),
    )


def expected_gain(gain, obs, x):
    """The gain of every row from the tracks of its antennas, linearly
    interpolated to its time, shape `(n_pol, n_rows, 1)`."""

    def at_rows(grid_model):
        tracks = np.asarray(grid_model(x))[..., 0]  # (n_pol, n_ant, n_grid)
        interp = lambda track: np.interp(obs.time, gain.time_grid, track)
        return np.apply_along_axis(interp, 2, tracks)

    rows = np.arange(obs.time.size)
    la, ph = at_rows(gain.log_amplitude), at_rows(gain.phase)
    a1, a2 = obs.ant1, obs.ant2
    log_amplitude = la[:, a1, rows] + la[:, a2, rows]
    phase = ph[:, a1, rows] - ph[:, a2, rows]
    return np.exp(log_amplitude + 1j * phase)[..., None]


def random_latent(domain, seed=0):
    return jft.random_like(jax.random.PRNGKey(seed), domain)


# --------------------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------------------


def test_calibration_block_parses():
    assert CONFIG == CalibrationConfig(
        dt=20.0,
        phase=TrackPrior((0.0, 0.5), (0.0, 0.0), (0.5, 0.2)),
        log_amplitude=TrackPrior((0.0, 0.1), (0.0, 0.0), (0.1, 0.05)),
    )
    assert CalibrationConfig.from_yaml_dict(None) is None


@pytest.mark.parametrize(
    "path,value,match",
    [
        (("log_amplitude",), None, r"missing \['log_amplitude'\]"),
        (("delay",), 1.0, r"unknown \['delay'\]"),
        (("phase", "curv"), [0.0, 1.0], r"calibration.phase.*unknown \['curv'\]"),
        (("phase", "slope"), None, r"calibration.phase.*missing \['slope'\]"),
        (("dt",), 0.0, "dt"),
        (("phase", "slope"), [0.0, -0.1], "calibration.phase.slope"),
        (("log_amplitude", "intercept"), [0.0], "calibration.log_amplitude.intercept"),
        (("phase", "deviations"), [0.5, 0.0], r"calibration.phase.deviations.*positive"),
        (
            ("log_amplitude", "deviations"),
            [0.0, 0.2],
            r"calibration.log_amplitude.deviations.*positive",
        ),
    ],
)
def test_invalid_calibration_block_raises(path, value, match):
    """`value` None deletes the key."""
    raw = deepcopy(CALIBRATION)
    block = raw if len(path) == 1 else raw[path[0]]
    if value is None:
        del block[path[-1]]
    else:
        block[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        CalibrationConfig.from_yaml_dict(raw)


# --------------------------------------------------------------------------------------
# Gain
# --------------------------------------------------------------------------------------


def test_gain_needs_the_config_and_the_antennas():
    obs = generate_random_obs(FREQS, 5, [-1e2, 1e2], [-5, 5], POL)
    assert build_gain(None, obs, "gain") is None
    with pytest.raises(ValueError, match="antennas"):
        build_gain(CONFIG, obs, "gain")


def test_gain_rejects_cross_hand_correlations():
    """RL needs the R gain of one antenna and the L gain of the other, which one
    track per correlation cannot give."""
    obs = gain_obs([0, 0, 1], [1, 2, 2], [0.0, 1.0, 2.0])
    full = rve.Observation(
        obs.antenna_positions,
        np.concatenate([obs.vis_val] * 2),
        np.concatenate([obs.weight_val] * 2),
        rve.Polarization((5, 6, 7, 8)),
        obs.freq,
        None,
    )
    with pytest.raises(ValueError, match="cross-hand"):
        build_gain(CONFIG, full, "gain")


def test_gain_is_one_at_the_prior_mean():
    """Intercept and slope means are 0 (the deviations are lognormal, their
    mean is positive), so the zero latent gives no gain."""
    obs = gain_obs([0, 0, 1], [1, 2, 2], [0.0, 1.3, 4.0])
    obs = rve.Observation(
        obs.antenna_positions,
        obs.vis_val.astype(np.complex64),
        obs.weight_val.astype(np.float32),
        obs.legacy_polarization,
        obs.freq,
        None,
    )
    gain = build_gain(CONFIG, obs, "gain")

    g = gain(jft.zeros_like(gain.domain))

    assert g.dtype == np.complex64
    assert g.shape == (2, 3, 1)
    assert_allclose(g, 1.0)


@pytest.mark.parametrize("ids", [(0, 1, 2), (0, 2, 5)], ids=["contiguous", "sparse"])
def test_gain_of_a_row_is_the_gain_of_its_antennas(ids):
    """Rows on the grid points: the gain is the tracks of its two antennas at
    that point. The ids index the tracks, so sparse ids need `max + 1` tracks."""
    a = np.array(ids)
    obs = gain_obs(a[[0, 0, 1, 0, 1, 2]], a[[1, 2, 2, 2, 2, 1]], [0, 0, 1, 2, 3, 3])
    gain = build_gain(CONFIG, obs, "gain")
    x = random_latent(gain.domain)

    assert gain.n_ant == ids[-1] + 1
    assert_allclose(gain.time_grid[:4], T0 + DT * np.arange(4))
    assert_allclose(gain(x), expected_gain(gain, obs, x), rtol=1e-12)


def test_rows_of_one_time_share_the_interpolated_tracks():
    """The tracks are interpolated once per time, `(n_pol, n_ant, n_times, 1)`."""
    steps = [1.5, 0.0, 1.5, 0.0, 3.2, 3.2]
    obs = gain_obs([0, 1, 0, 0, 1, 0], [1, 2, 2, 1, 2, 2], steps)
    gain = build_gain(CONFIG, obs, "gain")
    x = random_latent(gain.domain, seed=3)

    assert gain._gain._phases.target.shape == (2, 3, 3, 1)
    assert_allclose(gain(x), expected_gain(gain, obs, x), rtol=1e-12)


@pytest.mark.parametrize("last", [2.5, 3.0, 0.0], ids=["between", "on_grid", "single"])
def test_the_last_time_is_inside_the_grid(last):
    """Past the grid end the tracks interpolate to 0, a gain of exactly 1."""
    obs = gain_obs([0, 0, 1], [1, 2, 2], [0.0, last / 2, last])
    gain = build_gain(CONFIG, obs, "gain")
    x = random_latent(gain.domain, seed=1)

    g = gain(x)

    assert gain.time_grid[-1] > obs.time.max()
    assert_allclose(g, expected_gain(gain, obs, x), rtol=1e-12)
    assert np.all(np.abs(g[:, -1] - 1.0) > 1e-3)


def test_gain_latents_start_with_the_prefix():
    obs = gain_obs([0, 0, 1], [1, 2, 2], [0.0, 1.0, 2.0])
    a, b = (build_gain(CONFIG, obs, p) for p in ("gain_fld_0", "gain_fld_1"))

    assert set(a.domain) == {
        f"gain_fld_0_{kind}{suffix}"
        for kind in ("phase", "logamp")
        for suffix in ("", "_x0", "_sigma")
    }
    assert set(a.domain).isdisjoint(b.domain)


# --------------------------------------------------------------------------------------
# Likelihood
# --------------------------------------------------------------------------------------


def sky_grid(center=SkyCoord(ra=np.nan * u.rad, dec=np.nan * u.rad)):
    spatial = ju.wcs.WcsAstropy(center=center, shape=(32, 32), fov=[1.0, 1.5] * u.deg)
    return ju.Grid(
        spatial=spatial,
        spectral=ju.color.Color.from_central_frequencies(FREQS),
        polarization=ju.polarization.PolarizationType(("I",)),
    )


def band7_obs(n_rows=40):
    np.random.seed(45)
    ant1 = np.random.choice([0, 2, 5], n_rows)
    ant2 = np.random.choice([1, 3], n_rows)
    return gain_obs(ant1, ant2, np.random.uniform(0, 4, n_rows))


def band7(obs, config, backend="finufft", noise=None):
    """Builder of `obs` as one field, with the gain of `config`."""
    grid = sky_grid()
    sky_beamer = SkyBeamer(
        jft.ShapeWithDtype(grid.shape),
        {FIELD: BeamPattern(0.0, 0.0, np.ones(grid.shape), None)},
    )
    settings = {
        "ducc": rve.parse.Ducc0Settings(
            epsilon=1e-9, do_wgridding=False, nthreads=1, verbosity=False
        ),
        "finufft": rve.parse.FinufftSettings(epsilon=1e-9),
    }[backend]
    return build_likelihood_from_sky_beamer(
        observation=obs,
        field_name=FIELD,
        sky_beamer=sky_beamer,
        sky_grid=grid,
        backend_settings=settings,
        noise_std_correction=noise,
        gain=build_gain(config, obs, "gain"),
    )


def random_sky(seed=2):
    return {FIELD: np.random.default_rng(seed).normal(size=sky_grid().shape)}


def test_zero_gain_latent_leaves_the_likelihood_unchanged():
    obs = band7_obs()
    calibrated, plain = band7(obs, CONFIG), band7(obs, None)
    sky = random_sky()

    zero_gain = jft.zeros_like(calibrated.gain.domain)

    assert_allclose(
        calibrated.likelihood.energy(jft.Vector(sky | zero_gain)),
        plain.likelihood.energy(jft.Vector(sky)),
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    "noise", [None, BaseLineCorrection(alpha=2.0, scale=1.0)], ids=["fixed", "variable"]
)
def test_calibration_likelihood_is_the_likelihood_at_fixed_sky(noise):
    obs = band7_obs()
    calibrated, plain = band7(obs, CONFIG, noise=noise), band7(obs, None, noise=noise)
    sky = random_sky()

    likelihood = calibrated.calibration_likelihood(plain.response(sky))
    x = random_latent(likelihood.domain)

    assert isinstance(calibrated, VariableLikelihoodBuilder) == (noise is not None)
    assert likelihood.domain.tree.keys() == calibrated.likelihood.domain.tree.keys() - {
        FIELD
    }
    assert_allclose(
        likelihood.energy(x),
        calibrated.likelihood.energy(jft.Vector(sky | x.tree)),
        rtol=1e-12,
    )
    with pytest.raises(ValueError, match="gain"):
        plain.calibration_likelihood(plain.response(sky))


@pytest.mark.parametrize("backend", ("ducc", "finufft"))
def test_response_adjoint_holds_the_gain_fixed(backend):
    """<R_x s, d> == <s, R_x^H d> with the gain fixed at a sample x."""
    obs = band7_obs()
    calibrated, plain = band7(obs, CONFIG, backend), band7(obs, None, backend)
    sky = random_sky()
    x = random_latent(calibrated.gain.domain)
    d = calibrated.visibilities

    adjoint = calibrated.response_adjoint(d, x)[FIELD]

    lhs = np.real(np.sum(np.conj(calibrated.response(sky | x)) * d))
    assert_allclose(lhs, np.sum(sky[FIELD] * np.real(adjoint)), rtol=1e-6)
    # The gain reaches the adjoint, so it differs from the one without gain.
    uncalibrated = plain.response_adjoint(d)[FIELD]
    assert np.linalg.norm(adjoint - uncalibrated) > 0.1 * np.linalg.norm(uncalibrated)
    with pytest.raises(ValueError, match="gain"):
        calibrated.response_adjoint(d)


def test_radio_likelihood_gives_every_observation_its_own_gain(tmp_path):
    center = SkyCoord(ra=10.0 * u.deg, dec=20.0 * u.deg)
    pointing = np.array([center.ra.rad, center.dec.rad]).reshape(1, 1, 2)
    field = AuxiliaryTable(
        {"REFERENCE_DIR": pointing, "PHASE_DIR": pointing, "NAME": np.array(["fld"])}
    )
    for spw in (5, 7):
        obs = band7_obs()
        rve.Observation(
            obs.antenna_positions,
            obs.vis_val,
            obs.weight_val,
            obs.legacy_polarization,
            obs.freq,
            {"FIELD": field},
        ).save(str(tmp_path / f"obs_spw{spw}.npz"), False)
    data = {
        "data_path": str(tmp_path),
        "data_templates": ["obs_spw{spw}.npz"],
        "spectral": {"window": [5, 7]},
        "dish": {"size": 12.0, "blockage_size": 0.75},
    }
    grid = sky_grid(center)

    builders = {}
    for calibration in (None, CALIBRATION):
        response = {"backend": "finufft", "epsilon": 1e-9}
        if calibration is not None:
            response["calibration"] = calibration
        products = build_radio_likelihood(
            ["band7"],
            {"radio_response": response, "alma_data": {"band7": data}},
            sky_grid=grid,
            sky_domain={"sky": jft.ShapeWithDtype(grid.shape)},
            last_radio_bin=None,
        )
        likelihood = products.likelihood
        assert np.isfinite(likelihood.energy(random_latent(likelihood.domain)))
        builders[calibration is None] = products.likelihoods

    assert [b.gain for b in builders[True]] == [None, None]
    first, second = (b.gain for b in builders[False])
    assert set(first.domain).isdisjoint(second.domain)
    assert all(k.startswith("gain_") for g in (first, second) for k in g.domain)
