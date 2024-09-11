import nifty8.re as jft


def _model_wrap(model, target_domain=None):
    """ Wraps a model to ensure output consistency with the input domain. """
    if target_domain is None:
        def wrapper(x):
            out = model(x)
            for x, val in x.items():
                out[x] = val
            return out
    else:
        def wrapper(x):
            out = model(x)
            for key in target_domain.keys():
                out[key] = x[key]
            return out
    return wrapper


def connect_likelihood_to_model(
    likelihood: jft.Likelihood,
    model: jft.Model
) -> jft.Likelihood:
    """
    Connects a likelihood function to a model, updating the model's domain.

    This function is used when models are embedded within a likelihood.
    It ensures that the necessary model keys are propagated upwards through
    the chain, allowing white priors to be connected to the respective keys
    within the likelihood. The function updates the model's domain by
    merging it with the likelihood's domain and wraps the model for
    compatibility.

    Parameters
    ----------
    likelihood : jft.Likelihood
        The likelihood object that requires connection to the model.
    model : jft.Model
        The model to be connected to the likelihood.

    Returns
    -------
    jft.Likelihood
        The likelihood object updated with the model connection and the merged
        domain.
    """

    ldom = likelihood.domain.tree
    tdom = {t: ldom[t] for t in ldom.keys() if t not in model.target.keys()}
    mdom = tdom | model.domain

    model_wrapper = _model_wrap(model, tdom)
    model = jft.Model(
        lambda x: jft.Vector(model_wrapper(x)),
        domain=jft.Vector(mdom)
    )

    return likelihood.amend(model, domain=model.domain)


def build_gaussian_likelihood(
    data,
    std
):
    """
    Build a Gaussian likelihood function based on the provided data and
    standard deviation.

    This function creates a Gaussian likelihood for modeling data with
    Gaussian noise.
    It calculates the inverse variance and inverse standard deviation for
    noise handling and returns a `jft.Gaussian` likelihood object.

    Parameters
    ----------
    data : array-like
        The observed data to be modeled with a Gaussian likelihood.
    std : float or array-like
        The standard deviation of the noise.
        If a float is provided, a constant noise level is assumed for all
        data points. If an array is provided, it must have the
        same shape as `data`, representing a varying noise level.

    Returns
    -------
    jft.Gaussian
        A Gaussian likelihood object initialized with the provided data and
        noise parameters.

    Raises
    ------
    AssertionError
        If `std` is an array and its shape does not match the shape of `data`.
    """
    if not isinstance(std, float):
        assert data.shape == std.shape

    var_inv = 1/(std**2)
    std_inv = 1/std

    return jft.Gaussian(
        data=data,
        noise_cov_inv=lambda x: x*var_inv,
        noise_std_inv=lambda x: x*std_inv,
    )
