import nifty.re as jft
import jax.scipy as jsc
import jax.numpy as jnp

from jax.tree_util import Partial
from jax import vmap

from ..data.observation import Observation


class CalibrationInterpolator:
    """
    Interpolates visibilites for a specific sequence of antenna-time pairs given the visibilities
    on an evenly spaced antenna-time grid.

    Parameters
    ----------
    ant_col: jnp.ndarray
        Antenna points to which one wants to interpolate
    time_col: jnp.ndarry
        Time points to which one wants to interpolate
    dt: float
        Distances between time points on time axis.
    target_shape: tuple
        Shape of data in visibility space. Should in principle follow (n_correlation, n_baseline, n_freq).

    Note
    ----
    Currently, only uniformly spaced time axis is supported.
    """

    def __init__(
        self,
        ant_col: jnp.ndarray,
        time_col: jnp.ndarray,
        dt: float,
        target_shape: tuple,
    ):

        coords = [ant_col, time_col / dt]

        self._li = Partial(jsc.ndimage.map_coordinates, coordinates=coords, order=1)

        self._n_corr, _, self._n_freq = target_shape

    def __call__(self, x):
        res = vmap(
            vmap(
                lambda corr, freq: self._li(x[corr, :, :, freq]),
                in_axes=(None, 0),
                out_axes=1,
            ),
            in_axes=(0, None),
        )

        return res(jnp.arange(self._n_corr), jnp.arange(self._n_freq))


class CalibrationDistribution(jft.Model):
    """
    Computes the calibration operator from given observation data.

    Parameters
    ----------
    observation: Observation
        Observation object from which are the antenna and temporal information corresponding to
        the visibilites are extracted.
    phase_fields: jft.Model
        Model for phases of calibration solutions. Shape: (n_correlations, n_time, n_antennas, n_freq)
    log_amplitude_fields: jft.Model
        Model for log amplitude of calibration solutions. Shape: (n_correlations, n_time, n_antennas, n_freq)
    dt: float
        Distances between time points on time axis. Has to be the same distance of time points,
        which is used for phase_fields and log_amplitude fields.

    Note
    ----
    Currently, only uniformly spaced time axis are supported.
    """

    def __init__(
        self,
        observation: Observation,
        phase_fields: jft.Model,
        log_amplitude_fields: jft.Model,
        dt: float,
    ):
        ap = observation.antenna_positions
        target_shape = observation.vis.shape
        self._cop1 = CalibrationInterpolator(
            jnp.asarray(ap.ant1), jnp.asarray(ap.time), dt, target_shape
        )
        self._cop2 = CalibrationInterpolator(
            jnp.asarray(ap.ant2), jnp.asarray(ap.time), dt, target_shape
        )

        self._phases = phase_fields
        self._logamps = log_amplitude_fields

        super().__init__(init=self._phases.init | self._logamps.init)

    def __call__(self, x):
        res_logamp = jnp.real(
            self._cop1(self._logamps(x)) + self._cop2(self._logamps(x))
        )
        res_phase = (
            jnp.real(self._cop1(self._phases(x)) - self._cop2(self._phases(x))) * 1j
        )

        return jnp.exp(res_logamp + res_phase)
