from dataclasses import dataclass

import nifty.re as jft

from ..jwst_response import (
    JwstResponse,
)
from ..data.loader.target_loader import TargetDataCore
from .likelihood import (
    GaussianLikelihoodBuilder,
    VariableCovarianceGaussianLikelihoodBuilder,
)
from ..variable_covariance.inverse_standard_deviation import InverseStdBuilder

# --------------------------------------------------------------------------------------
# Jwst Likelihood
# --------------------------------------------------------------------------------------


# Output -------------------------------------------------------------------------------


@dataclass
class SingleTargetLikelihood:
    filter: str
    builder: GaussianLikelihoodBuilder | VariableCovarianceGaussianLikelihoodBuilder

    @property
    def likelihood(self) -> jft.Likelihood | jft.Gaussian:
        return self.builder.build()

    @property
    def response(self) -> JwstResponse:
        return self.builder.response


# Interface ----------------------------------------------------------------------------


def build_target_likelihood(
    filter_name: str,
    response: JwstResponse,
    target_data: TargetDataCore,
    inverse_std_builder: InverseStdBuilder | None = None,
):
    if inverse_std_builder is None:
        builder = GaussianLikelihoodBuilder(
            response=response,
            data=target_data.data,
            std=target_data.std,
            mask=target_data.mask,
        )

    else:
        builder = VariableCovarianceGaussianLikelihoodBuilder(
            response=response,
            inverse_std_builder=inverse_std_builder,
            data=target_data.data,
            std=target_data.std,
            mask=target_data.mask,
        )

    return SingleTargetLikelihood(filter=filter_name, builder=builder)
