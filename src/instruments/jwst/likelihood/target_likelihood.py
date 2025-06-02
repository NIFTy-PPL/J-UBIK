from dataclasses import dataclass

import nifty8.re as jft

from ..jwst_response import (
    JwstResponse,
    TargetResponseInput,
    build_target_response,
)
from ..plotting.residuals import ResidualPlottingInformation
from .likelihood import GaussianLikelihoodInput, LikelihoodData, build_likelihood

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
    builder: GaussianLikelihoodInput

    @property
    def likelihood(self) -> jft.Likelihood | jft.Gaussian:
        return build_likelihood(self.builder)


def build_target_likelihood(
    response: TargetResponseInput,
    side_effect: TargetLikelihoodSideEffects,
):
    jwst_target_response: JwstResponse = build_target_response(input_config=response)

    builder = GaussianLikelihoodInput(
        response=jwst_target_response,
        data=LikelihoodData.from_equivalent(response.target_data),
    )

    side_effect.plotting.append_information(
        filter=response.filter_name,
        data=response.target_data.data,
        std=response.target_data.std,
        mask=response.target_data.mask,
        model=jwst_target_response,
    )

    return SingleTargetLikelihood(filter=response.filter_name, builder=builder)
