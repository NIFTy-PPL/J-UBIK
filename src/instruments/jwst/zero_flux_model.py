# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %

import nifty8.re as jft

from .parametric_model import build_parametric_prior

ZERO_FLUX_KEY = "zero_flux"


def build_zero_flux_model(
    prefix: str,
    likelihood_config: dict,
) -> jft.Model:
    """
    Build a zero flux model based on the provided likelihood configuration.

    If no specific configuration for the zero-flux model is found in
    `likelihood_config`, a default model returning zero is created.
    Otherwise, it builds a parametric model based on the provided configuration.

    Parameters
    ----------
    prefix : str
        A string prefix used to identify and name the parameters associated with
         the zero-flux model.
    likelihood_config : dict
        A configuration dictionary containing model details.
        The zero-flux model configuration is expected to be under the key
        specified by `ZERO_FLUX_KEY`.

    Returns
    -------
    jft.Model
        A model representing the zero flux configuration.
        If no configuration is provided, the model returns zero;
        otherwise, it uses a parametric prior.
    """
    model_cfg = likelihood_config.get(ZERO_FLUX_KEY, None)
    if model_cfg is None:
        return jft.Model(lambda _: 0, domain=dict())

    prefix = "_".join([prefix, ZERO_FLUX_KEY])

    shape = (1,)
    prior = build_parametric_prior(prefix, model_cfg["prior"], shape)
    return jft.Model(prior, domain={prefix: jft.ShapeWithDtype(shape)})
