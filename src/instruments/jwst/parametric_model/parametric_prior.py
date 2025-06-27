# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %%
from ..parse.parametric_model.parametric_prior import ProbabilityConfig

from typing import Callable, Tuple, Union

import numpy as np
import jax.numpy as jnp
import nifty8.re as jft

DISTRIBUTION_MAPPING = {
    "normal": jft.normal_prior,
    "log_normal": jft.lognormal_prior,
    "lognormal": jft.lognormal_prior,
    "uniform": jft.uniform_prior,
    "invgamma": jft.invgamma_prior,
    "delta": lambda x: lambda _: x,
    None: lambda x: lambda _: x,
}


def _infer_shape(params: dict, shape: tuple):
    """Infers the shape of the prior."""
    if shape != ():
        return shape

    match params["distribution"]:
        case "delta" | None:
            return jnp.shape(params["mean"])

        case "uniform":
            shp1, shp2 = map(jnp.shape, (params["min"], params["max"]))

        case _:
            shp1, shp2 = map(jnp.shape, (params["mean"], params["sigma"]))

    # TODO: do some checks on compatibility of the two shapes
    return shp1


def _transform_setting(parameters: Union[dict, tuple]):
    """Transforms the prior setting into a dictionary."""
    if isinstance(parameters, dict):
        return parameters

    distribution = parameters[0].lower()

    match distribution:
        case "uniform":
            return dict(distribution=distribution, min=parameters[1], max=parameters[2])

        case "delta" | None:
            return dict(distribution=distribution, mean=parameters[1])

        case _:
            return dict(
                distribution=distribution, mean=parameters[1], sigma=parameters[2]
            )


def build_parametric_prior_from_prior_config(
    domain_key: str,
    prior_config: ProbabilityConfig,
    shape: Tuple[int] = (),
    as_model: bool = False,
) -> Union[Callable, jft.Model]:
    """
    Builds a parametric prior based on the specified distribution and
    transformation.

    This function constructs a prior distribution for a given model domain by
    interpreting the provided `prior_config` and selecting the appropriate
    prior function based on the distribution specified therein.
    The prior can be optionally transformed if a transformation
    function is specified.

    Parameters
    ----------
    domain_key : str
        A string key identifying the domain of the model to which this prior
        applies.
    prior_config : Union[DefaultPriorConfig, UniformPriorConfig, DeltaPriorConfig],
        A prior config containing the relevant parameters for the configuration
        of the probability distribution.
    shape : tuple of int, optional
        A tuple representing the shape of the parameters.
        This shape is applied to adjust the values of the prior to
        match the specified shape. Default is an empty tuple.

    Returns
    -------
    Callable
        A wrapped callable function representing the prior distribution.
        If a transformation is specified in `parameters`,
        the prior is wrapped with the transformation; otherwise,
        the raw prior function is returned.

    Raises
    ------
    NotImplementedError
        If the specified distribution is not found in the
        `DISTRIBUTION_MAPPING`.

    Example
    -------
    Given a Gaussian distribution with mean and standard deviation,
    the function returns a Gaussian prior with an optional transformation
    (e.g., `log`):

    >>> params = {'distribution': 'Gaussian', 'mean': 0, 'std': 1, 'transformation': 'log'}
    >>> prior = build_parametric_prior('my_domain', params)
    >>> prior(x)  # returns the log of the Gaussian prior evaluated at x

    Notes
    -----
    The available distributions and their required keys are stored in
    `DISTRIBUTION_MAPPING`, which maps each distribution to its corresponding
    prior function and required parameters.
    """

    distribution_builder = DISTRIBUTION_MAPPING[prior_config.distribution]
    distribution = distribution_builder(*prior_config.parameters_to_shape(shape=shape))

    if prior_config.transformation is not None:
        trafo = getattr(jnp, prior_config.transformation)
        func = jft.wrap(lambda x: trafo(distribution(x)), domain_key)
    else:
        func = jft.wrap(distribution, domain_key)

    if as_model:
        return jft.Model(func, domain={domain_key: jft.ShapeWithDtype(shape)})

    return func
