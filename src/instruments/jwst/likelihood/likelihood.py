from dataclasses import dataclass

import jax.numpy as jnp
import nifty8.re as jft
import numpy as np

from ....likelihood import build_gaussian_likelihood
from ..jwst_response import JwstResponse


# --------------------------------------------------------------------------------------
# Jwst Likelihood
# --------------------------------------------------------------------------------------


@dataclass
class GaussianLikelihoodBuilder:
    """Essential input data of `load_data`."""

    response: JwstResponse | jft.Model
    data: np.ndarray
    std: np.ndarray
    mask: np.ndarray

    def __post_init__(self):
        self.data = np.array(self.data)
        self.std = np.array(self.std)
        self.mask = np.array(self.mask)

    def build(self) -> jft.Gaussian:
        return build_gaussian_likelihood(
            jnp.array(np.array(self.data)[np.array(self.mask)], dtype=float),
            jnp.array(np.array(self.std)[np.array(self.mask)], dtype=float),
            model=self.response,
        )


@dataclass
class VariableCovarianceGaussianLikelihoodBuilder:
    """Essential input data of `load_data`."""

    response: JwstResponse | jft.Model
    std_inv: jft.Model
    data: np.ndarray
    std: np.ndarray
    mask: np.ndarray

    def __post_init__(self):
        self.data = np.array(self.data)
        self.std = np.array(self.std)
        self.mask = np.array(self.mask)

    def build(self) -> jft.VariableCovarianceGaussian:
        model = jft.Model(
            lambda x: (self.response(x), self.std_inv),
            domain=self.response.domain | self.std_inv.domain,
        )
        like = jft.VariableCovarianceGaussian(self.data[self.mask])
        return like.amend(model, domain=jft.Vector(model.domain))
