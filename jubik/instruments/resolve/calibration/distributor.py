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


class CalibrationInterpolator(jft.Model):
    """
    Interpolates calibration fields from a regular time or time-frequency grid
    onto the observation grid.

    The input calibration field is assumed to have shape

        (n_corr, n_ant, n_time, n_freq)

    where `n_freq` may either denote the observation frequency channels (time-only
    interpolation) or a separate calibration frequency grid (time and frequency
    interpolation).

    If only `time_col` is provided, interpolation is performed along the time axis,
    resulting in

        (n_corr, n_ant, n_time_obs, n_freq).

    If `freq_col` is additionally provided, interpolation is subsequently performed
    along the frequency axis, resulting in

        (n_corr, n_ant, n_time_obs, n_freq_obs).

    Parameters
    ----------
    time_col : jnp.ndarray
        Observation times for which calibration solutions are required.
    dt : float
        Spacing of the calibration time grid.
    n_corr : int
        Number of correlation products.
    n_ant : int
        Number of antennas.
    n_freq : int
        Number of frequency channels of the input calibration field. Required for
        time-only interpolation and as the input frequency dimension for
        time-frequency interpolation.
    freq_col : jnp.ndarray, optional
        Observation frequencies. If provided, interpolation is additionally
        performed along the frequency axis.
    df : float, optional
        Spacing of the calibration frequency grid. Required when `freq_col` is
        given.
    time_origin : float, optional
        Time represented by index zero of the calibration grid. Defaults to the
        first observation time, which makes absolute measurement-set timestamps
        safe to use.
    frequency_origin : float, optional
        Frequency represented by index zero of the calibration grid. Defaults to
        the first observation frequency.

    Notes
    -----
    Both time and frequency grids are assumed to be uniformly spaced. Time and
    frequency interpolation are performed sequentially using first-order linear
    interpolation.
    """

    def __init__(
        self,
        model: jft.Model,
        time_col: jnp.ndarray,
        dt: float,
        n_corr: int,
        n_ant: int,
        n_freq: int,
        freq_col: jnp.ndarray | None = None,
        df: float | None = None,
        time_origin: float | None = None,
        frequency_origin: float | None = None,
    ):

        if (n_freq is None) and (freq_col is None):
            raise ValueError(
                "Either set n_freq (for time only interpolation) or freq_col (for time and frequency interpolation)."
            )
        if dt <= 0:
            raise ValueError("The calibration time-grid spacing `dt` must be positive.")

        time_col = jnp.asarray(time_col)
        if time_col.size == 0:
            raise ValueError("Cannot interpolate calibration fields without times.")
        if time_origin is None:
            time_origin = time_col[0]
        li_time = Partial(
            map_coordinates,
            coordinates=[(time_col - time_origin) / dt],
            order=1,
        )

        # self._n_corr = n_corr
        # self._n_ant = n_ant

        if freq_col is None:
            self._interpolator = lambda x: interpolate_time(
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
            if df <= 0:
                raise ValueError(
                    "The calibration frequency-grid spacing `df` must be positive."
                )

            freq_col = jnp.asarray(freq_col)
            if freq_col.size == 0:
                raise ValueError(
                    "Cannot interpolate calibration fields without frequencies."
                )
            if frequency_origin is None:
                frequency_origin = freq_col[0]
            li_freq = Partial(
                map_coordinates,
                coordinates=[(freq_col - frequency_origin) / df],
                order=1,
            )

            self._interpolator = lambda x: interpolate_time_frequency(
                li_t=li_time,
                li_f=li_freq,
                x=x,
                n_corr=n_corr,
                n_ant=n_ant,
                n_freq_in=n_freq,
            )

        self._model = model

        super().__init__(init=model.init)

    def __call__(self, primals):
        return self._interpolator(self._model(primals))


class CalibrationDistributor(jft.Model):
    """
    Constructs the complex antenna-based calibration factors evaluated at the
    visibility sampling points of an observation.

    The supplied phase and log-amplitude models are assumed to produce calibration
    fields of shape

        (n_corr, n_ant, n_time_obs, n_freq_obs),

    where they are already interpolated onto the time and frequency points of the observation.

    For each visibility, the calibration is computed as

        exp(logamp_a1 + logamp_a2
            + 1j * (phase_a1 - phase_a2))

    yielding an output of shape

        (n_corr, n_vis, n_freq_obs)

    or

        (n_corr, n_vis, n_freq)

    if only time interpolation is performed.

    Parameters
    ----------
    observation : Observation
        Observation providing visibility sampling times, antenna indices and,
        optionally, observing frequencies.
    phase_fields : jft.Model
        Model returning the antenna-based phase calibration field.
    log_amplitude_fields : jft.Model
        Model returning the antenna-based log-amplitude calibration field.
    dt : float
        Spacing of the calibration time grid.
    frequency_grid : jnp.ndarray, optional
        Frequency grid on which the calibration fields are defined. If omitted,
        calibration fields are assumed to already be defined on the observation
        frequency channels and only time interpolation is performed.
    """

    def __init__(
        self,
        observation: Observation,
        phase_fields: jft.Model,
        log_amplitude_fields: jft.Model,
    ):

        self._gather_op = Partial(
            func=antenna_time_grid_to_data_point,
            time_col=jnp.asarray(observation.time),
            ant1_col=jnp.asarray(observation.ant1),
            ant2_col=jnp.asarray(observation.ant2),
        )

        self._phases = phase_fields
        self._logamps = log_amplitude_fields

        super().__init__(init=self._phases.init | self._logamps.init)

    def __call__(self, primals):
        res_logamp = self._gather_op(cube=self._logamps(primals), operation=jnp.add)
        res_phase = self._gather_op(cube=self._phases(primals), operation=jnp.subtract)

        return jnp.exp(res_logamp + 1j * res_phase)
