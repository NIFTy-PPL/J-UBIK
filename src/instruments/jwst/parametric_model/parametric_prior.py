# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %%
from ..parse.parametric_model.parametric_prior import ProbabilityConfig

from typing import Callable, Tuple, Union

import jax.numpy as jnp
import nifty8.re as jft

DISTRIBUTION_MAPPING = {
    'normal': (jft.normal_prior, ['mean', 'sigma']),
    'log_normal': (jft.lognormal_prior, ['mean', 'sigma']),
    'lognormal': (jft.lognormal_prior, ['mean', 'sigma']),
    'uniform': (jft.uniform_prior, ['min', 'max']),
    'delta': (lambda x: lambda _: x, ['mean']),
    None: (lambda x: lambda _: x, ['mean'])
}


def _shape_adjust(val, shape):
    """Adjusts the shape of the prior."""
    if jnp.shape(val) == shape:
        return jnp.array(val)
    else:
        return jnp.full(shape, val)


def _infer_shape(params: dict, shape: tuple):
    """Infers the shape of the prior."""
    if shape != ():
        return shape

    match params['distribution']:
        case 'delta' | None:
            return jnp.shape(params['mean'])

        case 'uniform':
            shp1, shp2 = map(jnp.shape, (params['min'], params['max']))

        case _:
            shp1, shp2 = map(jnp.shape, (params['mean'], params['sigma']))

    # TODO: do some checks on compatibility of the two shapes
    return shp1


def _transform_setting(parameters: Union[dict, tuple]):
    """Transforms the prior setting into a dictionary."""
    if isinstance(parameters, dict):
        return parameters

    distribution = parameters[0].lower()

    match distribution:
        case 'uniform':
            return dict(
                distribution=distribution,
                min=parameters[1],
                max=parameters[2])

        case 'delta' | None:
            return dict(
                distribution=distribution,
                mean=parameters[1]
            )

        case _:
            return dict(
                distribution=distribution,
                mean=parameters[1],
                sigma=parameters[2]
            )


def build_parametric_prior(
    domain_key: str,
    parameters: Union[dict, tuple],
    shape: Tuple[int] = ()
) -> Callable:
    """
    Builds a parametric prior based on the specified distribution and
    transformation.

    This function constructs a prior distribution for a given model domain by
    interpreting the provided `parameters` and selecting the appropriate
    prior function based on the distribution specified therein.
    The prior can be optionally transformed if a transformation
    function is specified.

    Parameters
    ----------
    domain_key : str
        A string key identifying the domain of the model to which this prior
        applies.
    parameters : dict or tuple
        A dictionary or tuple containing the prior configuration.
        The dictionary should have a 'distribution' key, which specifies the
        distribution type, and additional keys
        required for the distribution (e.g., 'mean', 'std' for a Gaussian).
        The required keys depend on the distribution.
    shape : tuple of int, optional
        A tuple representing the shape of the parameters.
        This shape is applied to adjust the values of the prior parameters to
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
    KeyError
        If the required keys for the distribution are missing in `parameters`.

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
    parameters = _transform_setting(parameters)

    try:
        distribution = parameters.get('distribution')
        prior_function, required_keys = DISTRIBUTION_MAPPING[distribution]
        vals = [_shape_adjust(parameters[key], shape) for key in required_keys]
        transformation = parameters.get('transformation', None)

    except KeyError as e:
        if distribution not in DISTRIBUTION_MAPPING:
            raise NotImplementedError(
                f"{domain_key}: Prior distribution '{distribution}' is not "
                "implemented. Available distributions: \n"
                f"{list(DISTRIBUTION_MAPPING.keys())}"
            ) from e
        else:
            raise KeyError(
                f"{domain_key}: The distribution '{distribution}' requires the"
                f" keys: {required_keys}"
            ) from e

    prior = prior_function(*vals)

    if transformation is not None:
        trafo = getattr(jnp, transformation)
        return jft.wrap(lambda x: trafo(prior(x)), domain_key)

    return jft.wrap(prior, domain_key)


def build_parametric_prior_from_prior_config(
    domain_key: str,
    prior_config: ProbabilityConfig,
    shape: Tuple[int] = ()
) -> Callable:
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

    distribution = prior_config.distribution
    prior_function, required_keys = DISTRIBUTION_MAPPING[distribution]
    vals = [_shape_adjust(getattr(prior_config, key), shape)
            for key in required_keys]
    transformation = prior_config.transformation

    prior = prior_function(*vals)

    if transformation is not None:
        trafo = getattr(jnp, transformation)
        return jft.wrap(lambda x: trafo(prior(x)), domain_key)

    return jft.wrap(prior, domain_key)
