# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %

import nifty.re as jft

from ...models.distributions import build_parametric_prior_from_prior_config
from ...parse.models.distributions import ProbabilityConfig

CALIBRATION_KEY = "calibration"


def build_calibration_model(
    prefix: str, prior_config: ProbabilityConfig | None, shape: tuple[int] = (1,)
) -> jft.Model | None:
    """
    Build a multiplicative calibration model, one factor per entry of `shape`.
    If `prior_config` is None, None is returned (no calibration, factor 1).

    Parameters
    ----------
    prefix : str
        A string prefix used for the prior domain of the calibration model.
    prior_config : ProbabilityConfig | None
        The prior of the calibration factor, e.g. a lognormal around 1.
    shape : tuple[int]
        The shape of the factors, e.g. `(n_exposures, 1, 1)`.
    """
    if prior_config is None:
        return None

    prefix = "_".join([prefix, CALIBRATION_KEY])
    return build_parametric_prior_from_prior_config(
        prefix, prior_config, shape, as_model=True
    )
