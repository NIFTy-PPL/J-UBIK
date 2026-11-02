"""Antenna gains of the model visibilities of one observation."""

from functools import partial

import nifty.re as jft
import numpy as np

from ..data.observation import Observation
from ..parse.calibration import CalibrationConfig, TrackPrior
from ..util import cast_to_dtype
from .distributor import CalibrationDistributor, CalibrationInterpolator
from .misc import CalibrationIWP, CalibrationStackedModel


class Gain(jft.Model):
    """Complex gain factor of the model visibilities, cast to `dtype`, see
    `build_gain`.

    Parameters
    ----------
    gain: jft.Model
        The gain of every row, shape `(n_pol, n_rows, 1)`.
    phase: jft.Model
        The phase tracks in rad on `time_grid`, shape `(n_pol, n_ant, n_grid, 1)`.
    log_amplitude: jft.Model
        The log-amplitude tracks on `time_grid`, shape `(n_pol, n_ant, n_grid, 1)`.
    time_grid: np.ndarray
        The times of the grid points, shape `(n_grid,)`, in the units of
        `Observation.time`.
    n_ant: int
        The number of antenna ids, the largest id plus one.
    dtype: np.dtype
        The dtype of the visibilities.
    """

    def __init__(
        self,
        gain: jft.Model,
        phase: jft.Model,
        log_amplitude: jft.Model,
        time_grid: np.ndarray,
        n_ant: int,
        dtype,
    ):
        self.phase = phase
        self.log_amplitude = log_amplitude
        self.time_grid = time_grid
        self.n_ant = n_ant
        self._gain = gain
        self._dtype = dtype
        super().__init__(init=gain.init)

    def __call__(self, x):
        return cast_to_dtype(self._gain(x), self._dtype)


def build_gain(
    config: CalibrationConfig | None, observation: Observation, prefix: str
) -> Gain | None:
    """Antenna gains of the model visibilities of one observation.

    For a row with antennas a1, a2 at time t and polarization p the gain is

        exp(logamp[p, a1, t] + logamp[p, a2, t]
            + 1j * (phase[p, a1, t] - phase[p, a2, t])),

    with every track linearly interpolated from a grid of spacing `config.dt`
    that starts at the first time of the observation.

    Parameters
    ----------
    config: CalibrationConfig | None
        The parsed `calibration` entry of `radio_response`; None means no gains.
    observation: Observation
        Provides the times, the antennas and the dtype of the visibilities.
    prefix: str
        Starts every latent key of the gain: `<prefix>_phase` and
        `<prefix>_logamp`, each with `_x0` and `_sigma`.

    Returns
    -------
    Gain | None
        The gain, shape `(n_pol, n_rows, 1)`, None without `config`.

    Raises
    ------
    ValueError
        If the observation has no antennas and times, or cross-hand
        correlations (RL, LR, XY, YX): one track per correlation cannot give
        the product of two hands.
    """
    if config is None:
        return None
    if observation.antenna_positions.only_imaging:
        raise ValueError(
            f"{prefix}: the gains need the antennas and times of the observation, "
            "which has only uvw"
        )
    if observation.legacy_polarization.has_crosshanded():
        raise ValueError(
            f"{prefix}: the gains are one track per parallel-hand correlation; a "
            "cross-hand product needs the gains of both hands, got "
            f"{observation.legacy_polarization.to_str_list()}"
        )
    # the raw antenna ids index the tracks, and they need not be contiguous
    n_ant = int(max(observation.ant1.max(), observation.ant2.max())) + 1
    t0 = observation.time.min()
    # one step past the last time: the interpolation is 0 outside the grid, a gain of 1
    n_steps = int(np.ceil((observation.time.max() - t0) / config.dt)) + 1

    def tracks(prior: TrackPrior, name: str) -> jft.Model:
        iwp = CalibrationIWP(
            n_steps,
            config.dt,
            *prior.intercept,
            *prior.slope,
            *prior.deviations,
            prefix=f"{prefix}_{name}",
        )
        # shortcut: frequency-flat gains, one track per (pol, antenna) for every
        # channel; frequency-dependent gains would go through the freq_col path
        # of CalibrationInterpolator
        column = jft.Model(lambda x: iwp(x)[:, None], init=iwp.init)
        return CalibrationStackedModel(column, n_corr=observation.npol, n_ant=n_ant)

    phase = tracks(config.phase, "phase")
    log_amplitude = tracks(config.log_amplitude, "logamp")
    # rows share times: interpolate once per time, not per row
    unique_times, time_index = np.unique(observation.time, return_inverse=True)
    interpolate = partial(
        CalibrationInterpolator,
        time_col=unique_times,
        dt=config.dt,
        n_corr=observation.npol,
        n_ant=n_ant,
        n_freq=1,
        time_origin=t0,
    )
    gain = CalibrationDistributor(
        observation, interpolate(phase), interpolate(log_amplitude), time_index
    )
    # the IWP has n_steps + 1 points
    time_grid = t0 + config.dt * np.arange(n_steps + 1)
    return Gain(
        gain, phase, log_amplitude, time_grid, n_ant, observation.vis_val.dtype
    )
