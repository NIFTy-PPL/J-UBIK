# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %

from .parametric_model import build_parametric_prior_from_prior_config
from .parse.parametric_model.parametric_prior import ProbabilityConfig

import nifty8.re as jft

from typing import Optional

ZERO_FLUX_KEY = 'zero_flux'
DEFAULT_KEY = 'default'
SHAPE = (1,)


def build_zero_flux_model(
    prefix: str,
    prior_config: Optional[ProbabilityConfig],
) -> Optional[jft.Model]:
    """
    Build a zero flux model based on the provided PriorConfig.
    If `prior_config` is None, None is returned.

    Parameters
    ----------
    prefix : str
        A string prefix used for the prior domain of the zero-flux model.
    prior_config : Optional[PriorConfig]
        The prior config, which is used to instantiate the prior probability 
        call.

    Returns
    -------
    Optional[jft.Model]
        A physical model representing the zero-flux of the observation.
        If `prior_config` is None, None is returned.
    """
    if prior_config is None:
        return None

    prefix = '_'.join([prefix, ZERO_FLUX_KEY])

    prior = build_parametric_prior_from_prior_config(
        prefix, prior_config, SHAPE)
    return jft.Model(prior, domain={prefix: jft.ShapeWithDtype(SHAPE)})
