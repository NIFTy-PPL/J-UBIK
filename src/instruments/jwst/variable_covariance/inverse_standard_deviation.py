from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from functools import reduce, partial
from typing import Any

import jax.numpy as jnp
import nifty8.re as jft
import numpy as np
from numpy.typing import ArrayLike

from ..data.loader.target_loader import TargetData
from ..parametric_model.parametric_prior import build_parametric_prior_from_prior_config
from ..parse.parametric_model.parametric_prior import ProbabilityConfig
from ..parse.variable_covariance import MultiplicativeStdValueConfig, StdValueShapeType

# ABC ----------------------------------------------------------------------------------


class InverseStdBuilder(ABC):
    @abstractmethod
    def build(self) -> jft.Model:
        pass

    @abstractmethod
    def update_fields(self, fields: dict) -> "InverseStdBuilder":
        """Build a new instance of the `InverseStdBuilder`, with updated fields."""
        pass


# Concrete Implementations -------------------------------------------------------------


@dataclass
class MultiplicativeStdValueBuilder(InverseStdBuilder):
    """Builder for a model, where the standard deviation is multiplied by a value.

    ...math : ``` inverse_std = 1/(std*value) ```
    """

    filter: str
    distribution: ProbabilityConfig
    shape_type: StdValueShapeType
    std: np.ndarray
    mask: np.ndarray

    def build(self) -> jft.Model:
        """Builds the model `1/(std*value)` from the fields."""
        distribution_builder = partial(
            build_parametric_prior_from_prior_config,
            domain_key=f"multistd_{self.filter}",
            prior_config=self.distribution,
            as_model=True,
        )
        one_over_std = 1 / self.std[self.mask]

        # Build concrete distribution & apply
        if self.shape_type == StdValueShapeType.filter:
            distribution = distribution_builder(shape=())

            def apply(x):
                val = 1 / distribution(x)
                return one_over_std * val

        elif self.shape_type == StdValueShapeType.pixel:
            distribution = distribution_builder(shape=(self.mask.sum(),))

            def apply(x):
                return one_over_std / distribution(x)

        elif self.shape_type == StdValueShapeType.integration:
            distribution = distribution_builder(shape=(self.std.shape[0],))
            sh = np.cumsum([0] + [m.sum() for m in self.mask])
            one_over_std = [
                one_over_std[sh[ii] : sh[ii + 1]] for ii in range(len(sh) - 1)
            ]

            def apply(x):
                val = 1 / distribution(x)
                one_over = [one_over_std[ii] * val[ii] for ii in range(len(val))]
                return reduce(jnp.append, one_over)

        else:
            raise ValueError(f"Unknown shape type: {self.shape_type}")

        return jft.Model(apply, domain=distribution.domain)

    def update_fields(self, new_fields: dict) -> "MultiplicativeStdValueBuilder":
        """Build a new instance of the `MultiplicativeStdValueBuilder`, with updated
        `new_fields`.

        Parameters
        ----------
        new_fields: dict
            A dictonary, where the key is the field to be udpated.
        """
        self_fields = {f.name: getattr(self, f.name) for f in fields(self)}
        self_fields.update(**new_fields)
        return MultiplicativeStdValueBuilder(**self_fields)


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
            distribution=config.distribution,
            shape_type=config.shape_type,
            std=target_data.std,
            mask=target_data.mask,
        )

    else:
        raise ValueError(f"Unknown config: {config}")
