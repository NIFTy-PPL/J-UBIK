from dataclasses import dataclass

import nifty8.re as jft
import numpy as np

from ..data.loader.target_loader import TargetData
from ..parse.parametric_model.parametric_prior import ProbabilityConfig
from ..parametric_model.parametric_prior import build_parametric_prior_from_prior_config
from ..parse.variable_covariance import MultiplicativeStdValueConfig

#


@dataclass
class MultiplicativeStdValueBuilder:
    filter: str
    value: ProbabilityConfig
    std: np.ndarray
    mask: np.ndarray

    def build(self) -> jft.Model:
        exit()
        value = build_parametric_prior_from_prior_config(
            f"multistd_{self.filter}",
            self.value,
            shape=(self.std.shape[0],),
            as_model=True,
        )
        one_over_std = 1 / self.std[self.mask]

        shapes = [m.sum() for m in self.mask]

        # def apply(x):
        #     val = value(x)
        #     return one_over_std[]*val
        #
        # return jft.Model(lambda x:


# API ----------------------------------------------------------------------------------


def build_inverse_standard_deviation(
    config: MultiplicativeStdValueConfig | None,
    filter_name: str,
    target_data: TargetData,
) -> jft.Model | None:
    if config is None:
        return None

    if isinstance(config, MultiplicativeStdValueConfig):
        return MultiplicativeStdValueBuilder(
            filter=filter_name,
            value=config.value,
            std=target_data.std,
            mask=target_data.mask,
        ).build()

    else:
        raise ValueError(f"Unknown config: {config}")
