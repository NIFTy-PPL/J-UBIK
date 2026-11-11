"""The Gaussian likelihood of one observation given predicted visibilities."""

import jax
import nifty.re as jft

from ..data.observation import Observation


def build_likelihood(
    observation: Observation,
    predicted_visibilities: jft.Model,
    inverse_std: jft.Model | None = None,
) -> jft.Likelihood:
    """The Gaussian likelihood of the visibilities of `observation`.

    The likelihood knows the data and the noise only: what predicts the
    visibilities (sky, shift, gains, or fixed model visibilities times gains)
    is the caller's `predicted_visibilities` model.
    Flagged points (weight 0) are left out.

    Parameters
    ----------
    observation: Observation
        The observed visibilities and their weights.
    predicted_visibilities: jft.Model
        The predicted visibilities, shape of `observation.vis`.
    inverse_std: jft.Model | None
        The inferred inverse noise standard deviation per real and imaginary
        part, shape of `observation.vis`, see `factory_noise_correction_model`;
        None uses the weights of `observation`. With it the energy has the
        normalization `-2 sum(log inverse_std)`: two real numbers per visibility.

    Raises
    ------
    ValueError
        If `predicted_visibilities` or `inverse_std` does not have the shape
        of the data.

    Examples
    --------
    The sky alone, a bare sky array as input:

    >>> R = interferometry_response(obs, grid, backend, beam=beam)
    >>> lh = build_likelihood(obs, R).amend(sky_model)

    Combining models (a noise model, gains, several observations, a sky model
    connected once) needs named latents, e.g. `sky_key="sky"`:

    >>> R = interferometry_response(obs, grid, backend, beam=beam, sky_key="sky")

    A gain calibration step, the sky fixed at model visibilities computed
    once:

    >>> v_model = R({"sky": sky_mean})
    >>> vis = jft.Model(lambda x: gains(x) * v_model, domain=gains.domain)
    >>> lh = build_likelihood(obs, vis)

    A self-calibration step, the sky learned alongside the gains; the sky
    model (latents to `{"sky": sky}`) is connected once, after the likelihoods
    of all observations are summed:

    >>> vis = jft.Model(lambda x: gains(x) * R(x), domain=R.domain | gains.domain)
    >>> lh = build_likelihood(obs, vis)
    >>> lh = connect_likelihood_to_model(lh, sky_model)
    """
    data = observation.vis
    predicted = tuple(predicted_visibilities.target.shape)
    if predicted != data.shape:
        raise ValueError(f"predicted visibilities {predicted}, data {data.shape}")
    # flagged points (weight 0) may hold NaN visibilities, so leave them out
    good = observation.mask_val
    data = data[good]

    if inverse_std is None:
        weight = observation.weight[good]
        likelihood = jft.Gaussian(data, noise_cov_inv=lambda x: x * weight)
        domain = predicted_visibilities.domain
        model = jft.Model(
            lambda x: predicted_visibilities(x)[good], init=predicted_visibilities.init
        )
        # named latents as a `Vector`, so likelihoods sum and connect to models
        domain = jft.Vector(domain) if isinstance(domain, dict) else domain
        return likelihood.amend(model, domain=domain)

    # a broadcast inverse_std would count its log normalization once, not per visibility
    noise = tuple(inverse_std.target.shape)
    if noise != observation.vis.shape:
        raise ValueError(f"inverse_std {noise}, data {observation.vis.shape}")

    def init(key):  # nifty's `|` of initializers fails on callable ones
        key_vis, key_noise = jax.random.split(key)
        return predicted_visibilities.init(key_vis) | inverse_std.init(key_noise)

    model = jft.Model(
        lambda x: (predicted_visibilities(x)[good], inverse_std(x)[good]),
        domain=predicted_visibilities.domain | inverse_std.domain,
        init=init,
    )
    likelihood = jft.VariableCovarianceGaussian(data)
    return likelihood.amend(model, domain=jft.Vector(model.domain))
