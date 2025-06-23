from dataclasses import dataclass, fields, is_dataclass
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

    @abstractmethod
    def update_fields(self, fields: dict) -> "InverseStdBuilder":
        """Build a new instance of the `InverseStdBuilder`, with updated fields."""
        pass


# Concrete Implementations -------------------------------------------------------------


@dataclass
class MultiplicativeStdValueBuilder(InverseStdBuilder):
    filter: str
    value: ProbabilityConfig
    std: np.ndarray
    mask: np.ndarray

    def build(self) -> jft.Model:
        """Build the InverseStandardDeviationModel from the fields."""

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
            val = 1 / value(x)
            one_over = [one_over_std[ii] * val[ii] for ii in range(len(val))]
            return reduce(jnp.append, one_over)

        return jft.Model(apply, domain=value.domain)

    def update_fields(self, fields: dict) -> "MultiplicativeStdValueBuilder":
        """Build a new instance of the `MultiplicativeStdValueBuilder`, with updated fields."""
        self_fields: dict = shallow_asdict(self)
        self_fields.update(**fields)
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
            value=config.value,
            std=target_data.std,
            mask=target_data.mask,
        )

    else:
        raise ValueError(f"Unknown config: {config}")


# Helper function ----------------------------------------------------------------------


def shallow_asdict(obj):
    """
    Returns a shallow dictionary representation of a dataclass instance,
    without recursing into nested dataclasses.
    """
    if not is_dataclass(obj):
        raise TypeError("shallow_asdict() should be called on dataclass instances")

    # Create a new dictionary from the dataclass fields
    return {f.name: getattr(obj, f.name) for f in fields(obj)}
