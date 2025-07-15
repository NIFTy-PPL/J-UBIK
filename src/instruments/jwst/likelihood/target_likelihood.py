from dataclasses import dataclass

import nifty.re as jft

from ..jwst_response import (
    JwstResponse,
)
from ..data.loader.target_loader import TargetDataCore
from ..plotting.residuals import ResidualPlottingInformation
from .likelihood import (
    GaussianLikelihoodBuilder,
    VariableCovarianceGaussianLikelihoodBuilder,
)
from ..variable_covariance.inverse_standard_deviation import InverseStdBuilder

# --------------------------------------------------------------------------------------
# Jwst Likelihood
# --------------------------------------------------------------------------------------


# Input --------------------------------------------------------------------------------


@dataclass
class TargetLikelihoodSideEffects:
    plotting: ResidualPlottingInformation


# Output -------------------------------------------------------------------------------


@dataclass
class SingleTargetLikelihood:
    filter: str
    builder: GaussianLikelihoodBuilder | VariableCovarianceGaussianLikelihoodBuilder

    @property
    def likelihood(self) -> jft.Likelihood | jft.Gaussian:
        return self.builder.build()


# Interface ----------------------------------------------------------------------------


def build_target_likelihood(
    response: JwstResponse,
    target_data: TargetDataCore,
    filter_name: str,
    inverse_std_builder: InverseStdBuilder | None = None,
    side_effect: TargetLikelihoodSideEffects | None = None,
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

    if side_effect is not None:
        side_effect.plotting.append_information(
            filter=filter_name,
            data=target_data.data,
            std=target_data.std,
            mask=target_data.mask,
            builder=builder,
        )

    return SingleTargetLikelihood(filter=filter_name, builder=builder)
