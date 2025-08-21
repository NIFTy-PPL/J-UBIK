from ...grid import Grid
from ...instruments.jwst.parse.parametric_model.parametric_prior import (
    transform_setting_to_prior_config,
)
from ...instruments.jwst.parametric_model.parametric_prior import (
    build_parametric_prior_from_prior_config,
)

import nifty.re as jft

import jax.numpy as jnp


class SingleValueMf(jft.Model):
    def __init__(self, value: jft.Model, shape: tuple[int]):
        self.value = value
        self._shape = shape
        super().__init__(domain=value.domain)

    def __call__(self, x):
        return jnp.full(self._shape, self.value(x))


def build_constant_mf_from_grid(
    grid: Grid,
    prefix: str,
    constant_cfg: dict,
):
    VALUE_KEY = "value"

    domkey = f"{prefix}_constant"
    value_distribution = build_parametric_prior_from_prior_config(
        domain_key=domkey,
        prior_config=transform_setting_to_prior_config(constant_cfg[VALUE_KEY]),
        as_model=True,
    )

    # More memory efficient & performant than saving a full array.
    shape = grid.shape[:-2] + (1, 1)

    return SingleValueMf(value=value_distribution, shape=shape)
