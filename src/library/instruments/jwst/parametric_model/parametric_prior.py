import nifty8.re as jft
import jax.numpy as jnp

from typing import Callable, Tuple, Union

DISTRIBUTION_MAPPING = {
    'normal': (jft.normal_prior, ['mean', 'sigma']),
    'log_normal': (jft.lognormal_prior, ['mean', 'sigma']),
    'lognormal': (jft.lognormal_prior, ['mean', 'sigma']),
    'uniform': (jft.uniform_prior, ['min', 'max']),
    'delta': (lambda x: lambda _: x, ['mean']),
    None: (lambda x: lambda _: x, ['mean'])
}


def _shape_adjust(val, shape):
    if jnp.shape(val) == shape:
        return jnp.array(val)
    else:
        return jnp.full(shape, val)


def _infer_shape(params: dict, shape: tuple):
    if shape != ():
        return shape

    match params['distribution']:
        case 'delta' | None:
            return jnp.shape(params['mean'])

        case 'uniform':
            shp1, shp2 = map(jnp.shape, (params['min'], params['max']))

        case _:
            shp1, shp2 = map(jnp.shape, (params['mean'], params['sigma']))

    # FIXME do some checks on compatibility of the two shapes
    return shp1


def _transform_setting(parameters: Union[dict, tuple]):
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
    domain_key: str, parameters: Union[dict, tuple], shape: Tuple[int] = ()
) -> Callable:

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


class ParametricPrior(jft.Model):
    def __init__(self, priors: dict, domain: dict):
        self._keys = priors.keys()
        self._priors = priors

        super().__init__(domain=domain)

    def __call__(self, x):
        return {key: self._priors[key](x) for key in self._keys}


def build_parametric_prior_model(
    domain_key: str,
    parameters: dict,
) -> jft.Model:

    ptree = {}
    priors = {}
    for key, params in parameters.items():
        dom_key = '_'.join((domain_key, key))
        params = _transform_setting(params)
        shape = _infer_shape(params)

        ptree[key] = jft.ShapeWithDtype(shape)
        priors[key] = build_parametric_prior(dom_key, params, shape)

    return ParametricPrior(priors, ptree)
