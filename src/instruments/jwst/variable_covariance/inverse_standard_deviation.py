from dataclasses import dataclass
from functools import reduce
from abc import ABC, abstractmethod

import nifty8.re as jft
import numpy as np
import jax.numpy as jnp

from ..data.loader.target_loader import TargetData
from ..parse.parametric_model.parametric_prior import ProbabilityConfig
from ..parametric_model.parametric_prior import build_parametric_prior_from_prior_config
from ..parse.variable_covariance import MultiplicativeStdValueConfig

# ABC ----------------------------------------------------------------------------------


class InverseStdBuilder(ABC):
    @abstractmethod
    def build(self) -> jft.Model:
        pass


# Concrete Implementations -------------------------------------------------------------


@dataclass
class MultiplicativeStdValueBuilder(InverseStdBuilder):
    filter: str
    value: ProbabilityConfig
    std: np.ndarray
    mask: np.ndarray

    def build(self) -> jft.Model:
        value = build_parametric_prior_from_prior_config(
            f"multistd_{self.filter}",
            self.value,
            shape=(self.std.shape[0],),
            as_model=True,
        )

        sh = np.cumsum([0] + [m.sum() for m in self.mask])
        one_over_std = 1 / self.std[self.mask]
        one_over_std = [one_over_std[sh[ii] : sh[ii + 1]] for ii in range(len(sh) - 1)]

        def apply(x):
            val = value(x)
            one_over = [one_over_std[ii] * val[ii] for ii in range(len(val))]
            return reduce(jnp.append, one_over)

        return jft.Model(apply, domain=value.domain)


# API ----------------------------------------------------------------------------------


def build_inverse_standard_deviation(
    config: MultiplicativeStdValueConfig | None,
    filter_name: str,
    target_data: TargetData,
) -> MultiplicativeStdValueBuilder | None:
    if config is None:
        return None

    if isinstance(config, MultiplicativeStdValueConfig):
        return MultiplicativeStdValueBuilder(
            filter=filter_name,
            value=config.value,
            std=target_data.std,
            mask=target_data.mask,
        )

    else:
        raise ValueError(f"Unknown config: {config}")
