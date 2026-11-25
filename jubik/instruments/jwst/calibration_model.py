# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %

import jax.numpy as jnp
import nifty.re as jft

from ...models.distributions import build_parametric_prior_from_prior_config
from ...parse.models.distributions import ProbabilityConfig

CALIBRATION_KEY = "calibration"


def build_calibration_model(
    prefix: str,
    prior_config: ProbabilityConfig | None,
    shape: tuple[int] = (1,),
    relative: bool = False,
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
    relative : bool
        Divide the factors by their geometric mean, mean-zero log factors: a
        factor common to all of them is degenerate with the sky they multiply.
        Needs a lognormal prior without a transformation, for positive factors;
        only the prior's width matters, its mean drops out.
    """
    if prior_config is None:
        return None
    lognormal = prior_config.distribution in ("lognormal", "log_normal")
    if relative and not (lognormal and prior_config.transformation is None):
        raise ValueError(
            "calibration factors relative to their geometric mean need a lognormal "
            f"prior without a transformation, got {prior_config.distribution} "
            f"with transformation {prior_config.transformation}"
        )

    prefix = "_".join([prefix, CALIBRATION_KEY])
    model = build_parametric_prior_from_prior_config(
        prefix, prior_config, shape, as_model=True
    )
    if not relative:
        return model

    def factors(x):
        f = model(x)
        return f / jnp.exp(jnp.mean(jnp.log(f)))

    return jft.Model(factors, domain=model.domain)
