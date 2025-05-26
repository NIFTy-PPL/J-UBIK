from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ....likelihood import build_gaussian_likelihood
from ..data.loader.target_loader import TargetData
from ..data.loader.stars_loader import SingleStarDataStacked
from ..jwst_response import JwstResponse


# --------------------------------------------------------------------------------------
# Jwst Likelihood
# --------------------------------------------------------------------------------------


@dataclass
class GaussianLikelihoodInput:
    """Essential input data of `load_data`."""

    response: JwstResponse
    data: TargetData | SingleStarDataStacked


def build_likelihood(input: GaussianLikelihoodInput):
    data = input.data

    return build_gaussian_likelihood(
        jnp.array(np.array(data.data)[np.array(data.mask)], dtype=float),
        jnp.array(np.array(data.std)[np.array(data.mask)], dtype=float),
        model=input.response,
    )
