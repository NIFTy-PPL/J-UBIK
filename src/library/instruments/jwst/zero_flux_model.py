import nifty8.re as jft

from .parametric_model import build_parametric_prior

ZERO_FLUX_KEY = 'zero_flux'


def build_zero_flux_model(
    prefix: str,
    likelihood_config: dict,
) -> jft.Model:
    model_cfg = likelihood_config.get(ZERO_FLUX_KEY, None)
    if model_cfg is None:
        return jft.Model(lambda _: 0, domain=dict())

    prefix = '_'.join([prefix, ZERO_FLUX_KEY])

    shape = (1,)
    prior = build_parametric_prior(prefix, model_cfg['prior'], shape)
    return jft.Model(prior, domain={prefix: jft.ShapeWithDtype(shape)})
