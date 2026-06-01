import operator
from functools import reduce

import jax.numpy as jnp
import nifty.re as jft

from ...grid import Grid
from .multifrequency_model import build_multifrequency_from_grid


class MfVstack(jft.Model):
    def __init__(self, models: dict):
        for key, model in models.items():
            setattr(self, key, model)

        self._models = [m for m in models.values()]

        super().__init__(
            domain=reduce(operator.or_, [m.domain for m in models.values()])
        )

    def __call__(self, x):
        m = [m(x) for m in self._models]
        return jnp.vstack(m)


def build_mfvstack_from_yaml_dict(
    grid: Grid,
    spectral_slices: list[tuple[int | None]],
    vstack_config: dict,
    prefix: str,
    **kwargs: dict,
):
    VSTACK_ORDER = "vstack_order"

    stacking_keys = vstack_config[VSTACK_ORDER]

    models = {}
    for skey, spsl in zip(stacking_keys, spectral_slices):
        skey_grid = Grid(
            spatial=grid.spatial, spectral=grid.spectral[spsl[0] : spsl[1]]
        )

        model = build_multifrequency_from_grid(
            grid=skey_grid,
            prefix=f"{prefix}_{skey}",
            model_cfg=vstack_config[skey],
            **kwargs,
        )
        models[skey] = model

    return MfVstack(models)
