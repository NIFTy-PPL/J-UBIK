from dataclasses import dataclass

import jax.numpy as jnp
import nifty8.re as jft
import numpy as np

from ....likelihood import build_gaussian_likelihood
from ..data.loader.target_loader import TargetData
from ..data.loader.stars_loader import SingleStarDataStacked
from ..jwst_response import JwstResponse


# --------------------------------------------------------------------------------------
# Jwst Likelihood
# --------------------------------------------------------------------------------------


@dataclass
class LikelihoodData:
    data: np.ndarray
    std: np.ndarray
    mask: np.ndarray

    def __post_init__(self):
        self.data = np.array(self.data)
        self.std = np.array(self.std)
        self.mask = np.array(self.mask)

    @classmethod
    def from_equivalent(cls, container: TargetData | SingleStarDataStacked):
        return cls(data=container.data, std=container.std, mask=container.mask)


@dataclass
class GaussianLikelihoodInput:
    """Essential input data of `load_data`."""

    response: JwstResponse
    data: LikelihoodData


def build_likelihood(input: GaussianLikelihoodInput) -> jft.Gaussian:
    data = input.data

    if isinstance(input, GaussianLikelihoodInput):
        return build_gaussian_likelihood(
            jnp.array(np.array(data.data)[np.array(data.mask)], dtype=float),
            jnp.array(np.array(data.std)[np.array(data.mask)], dtype=float),
            model=input.response,
        )
    else:
        NotImplementedError(f"{input} is unknown.")
