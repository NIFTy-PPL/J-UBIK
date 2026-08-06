import nifty.re as jft
import jax.numpy as jnp


class CalibrationStackedModel(jft.Model):
    """
    Vectorizes a model over data correlation, antenna and,
    optionally, frequency dimensions if `n_freq` is set.

    When `n_freq` is set the class assumes that the model is
    one-dimensional with output shape (n_time,).
    When `n_freq` is not set the class assumes that the model is
    two-dimensional with output shape (n_time, n_freq).

    The general output shape of the staked model is

        (n_corr, n_ant, n_time, n_freq).

    Parameters
    ----------
    model : jft.Model
        Base calibration field model. If `n_freq` is given, the model is
        assumed to be one-dimensional in time. Otherwise, it is assumed to
        already describe both time and frequency.
    n_corr : int
        Number of correlation products.
    n_ant : int
        Number of antennas.
    n_freq : int, optional
        Number of frequency channels. If provided, the base model is
        replicated independently along a newly introduced frequency axis.
        If omitted, the base model is assumed to already include the
        frequency dimension.
    """

    def __init__(
        self,
        model: jft.Model,
        n_corr: int,
        n_ant: int,
        n_freq: int | None = None,
    ):
        model_f = (
            model if n_freq is None else jft.VModel(model, axis_size=n_freq, out_axes=1)
        )
        model_a_f = jft.VModel(model_f, axis_size=n_ant, out_axes=0)
        self._model_c_a_f = jft.VModel(model_a_f, axis_size=n_corr, out_axes=0)

        super().__init__(init=self._model_c_a_f.init)

    def __call__(self, primals):
        return self._model_c_a_f(primals)


class CalibrationIWP(jft.Model):
    """
    One-dimensional field based on an integrated Wiener process.

    The model describes a linear trend with smooth non-parametric deviations.
    The linear component is parameterized by an intercept and a slope, while
    the stochastic component is given by an integrated Wiener process,
    resulting in realizations that are continuously differentiable.

    The strength of the stochastic component is rescaled such that the
    `deviations_mean` and `deviations_std` hyperparameters specify the
    expected magnitude of the deviations over the full extent of the process
    rather than per unit interval.

    Parameters
    ----------
    N_bins : int
        Number of bins of the discretized process.
    bin_width : float
        Width of each bin.
    intercepts_mean : float
        Prior mean of the intercept.
    intercepts_std : float
        Prior standard deviation of the intercept.
    slopes_mean : float
        Prior mean of the slope.
    slopes_std : float
        Prior standard deviation of the slope.
    deviations_mean : float
        Prior mean of the characteristic deviation from the linear trend over
        the full range of the process.
    deviations_std : float
        Prior standard deviation of the characteristic deviation from the
        linear trend over the full range of the process.
    prefix : str, optional
        Prefix used for naming the latent parameters of the model.

    Notes
    -----
    Let ``vol = N_bins * bin_width`` denote the total extent of the process.
    The stochastic component is rescaled by ``3 / vol**3`` so that the prior on
    the deviations is specified over the entire interval ``[0, vol]`` instead
    of depending on the discretization or total process length.
    """

    def __init__(
        self,
        N_bins,
        bin_width,
        intercepts_mean,
        intercepts_std,
        slopes_mean,
        slopes_std,
        deviations_mean,
        deviations_std,
        prefix="iwp",
    ):
        vol = N_bins * bin_width
        # Needed such that expected deviations can be set over the whole process
        rescale_factor = 3 / vol**3
        self._iwp = jft.IntegratedWienerProcess(
            x0=(
                jnp.array([intercepts_mean, slopes_mean]),
                jnp.array([intercepts_std, slopes_std]),
            ),
            sigma=(deviations_mean * rescale_factor, deviations_std * rescale_factor),
            dt=bin_width,
            N_steps=N_bins,
            name=prefix,
        )

        super().__init__(init=self._iwp.init)

    def __call__(self, primals):
        return self._iwp(primals)[:, 0]
