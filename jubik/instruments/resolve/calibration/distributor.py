import nifty.re as jft
import jax.numpy as jnp

from jax.tree_util import Partial
from jax import vmap
from jax.scipy.ndimage import map_coordinates

from ..data.observation import Observation, unique_antennas


def antenna_time_grid_to_data_point(cube, time_col, ant1_col, ant2_col, operation):
    def single_baseline_op(t, a1, a2):
        return operation(cube[:, a1, t, :], cube[:, a2, t, :])

    return vmap(single_baseline_op, out_axes=1)(
        jnp.arange(time_col.size), ant1_col, ant2_col
    )


def interpolate_time(li, x, n_corr, n_ant, n_freq):
    res = vmap(
        # Vmaps over correlation axis
        fun=vmap(
            # Vmaps over antenna axis
            fun=vmap(
                # Vmaps over frequency axis
                fun=lambda corr, ant, freq: li(x[corr, ant, :, freq]),
                in_axes=(None, None, 0),
                out_axes=1,
            ),
            in_axes=(None, 0, None),
            out_axes=0,
        ),
        in_axes=(0, None, None),
        out_axes=0,
    )

    return res(jnp.arange(n_corr), jnp.arange(n_ant), jnp.arange(n_freq))


def interpolate_frequency(li, x, n_corr, n_ant, n_time):
    res = vmap(
        # Vmaps over correlation axis
        fun=vmap(
            # Vmaps over antenna axis
            fun=vmap(
                # Vmaps over time axis
                fun=lambda corr, ant, time: li(x[corr, ant, time, :]),
                in_axes=(None, None, 0),
                out_axes=0,
            ),
            in_axes=(None, 0, None),
            out_axes=0,
        ),
        in_axes=(0, None, None),
        out_axes=0,
    )

    return res(jnp.arange(n_corr), jnp.arange(n_ant), jnp.arange(n_time))


def interpolate_time_frequency(li_t, li_f, x, n_corr, n_ant, n_freq_in):
    t_interp_time = interpolate_time(
        li=li_t,
        x=x,
        n_corr=n_corr,
        n_ant=n_ant,
        n_freq=n_freq_in,
    )

    return interpolate_frequency(
        li=li_f,
        x=t_interp_time,
        n_corr=n_corr,
        n_ant=n_ant,
        n_time=t_interp_time.shape[2],
    )


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
        time_col: jnp.ndarray,
        dt: float,
        n_corr: int,
        n_ant: int,
        n_freq: int | None = None,
        freq_col: jnp.ndarray | None = None,
        df: float | None = None,
    ):

        if (n_freq is None) and (freq_col is None):
            raise ValueError(
                "Either set n_freq (for time only interpolation) or freq_col (for time and frequency interpolation)."
            )
        li_time = Partial(map_coordinates, coordinates=[time_col / dt], order=1)

        # self._n_corr = n_corr
        # self._n_ant = n_ant

        if freq_col is None:
            self._call = lambda x: interpolate_time(
                li=li_time,
                n_corr=n_corr,
                x=x,
                n_ant=n_ant,
                n_freq=n_freq,
            )
        else:
            if df is None:
                raise ValueError(
                    "Set frequency bin width of time and frequency interpolation."
                )

            li_freq = Partial(map_coordinates, coordinates=[freq_col / df], order=1)

            self._call = lambda x: interpolate_time_frequency(
                li_t=li_time,
                li_f=li_freq,
                x=x,
                n_corr=n_corr,
                n_ant=n_ant,
                n_freq_in=n_freq,
            )

    def __call__(self, x):
        return self._call(x)


class CalibrationDistributor(jft.Model):
    """
    Computes the calibration operator from given observation data.

    Parameters
    ----------
    observation: Observation
        Observation object from which are the antenna and temporal information corresponding to
        the visibilites are extracted.
    phase_fields: jft.Model
        Model for phases of calibration solutions. Shape: (n_correlations, n_antennas, n_time, n_freq)
    log_amplitude_fields: jft.Model
        Model for log amplitude of calibration solutions. Shape: (n_correlations, n_antennas, n_time, n_freq)
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
        frequency_grid: jnp.ndarray | None = None,
    ):
        time = jnp.asarray(observation.time)

        n_corr, _, n_freq = observation.vis_val.shape
        n_ant = unique_antennas(observation)

        if frequency_grid is None:
            freq_col = None
            df = None
        else:
            freq_col = observation.freq
            df = jnp.diff(frequency_grid)[0]
            n_freq = frequency_grid.size

        self._interpolator = CalibrationInterpolator(
            time_col=observation.time,
            dt=dt,
            n_corr=n_corr,
            n_ant=n_ant,
            n_freq=n_freq,
            freq_col=freq_col,
            df=df,
        )

        self._gather_op = Partial(
            func=antenna_time_grid_to_data_point,
            time_col=time,
            ant1_col=jnp.asarray(observation.ant1),
            ant2_col=jnp.asarray(observation.ant2),
        )

        self._phases = phase_fields
        self._logamps = log_amplitude_fields

        super().__init__(init=self._phases.init | self._logamps.init)

    def __call__(self, primals):
        logamps_interp = self._interpolator(self._logamps(primals))
        phases_interp = self._interpolator(self._phases(primals))

        res_logamp = self._gather_op(logamps_interp, jnp.add)
        res_phase = self._gather_op(phases_interp, jnp.subtract)

        return jnp.exp(res_logamp + 1j * res_phase)
