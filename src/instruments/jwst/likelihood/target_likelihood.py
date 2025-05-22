from dataclasses import dataclass

from ..jwst_response import (
    JwstResponse,
    TargetResponseInput,
    build_target_response,
)
from ..plotting.residuals import ResidualPlottingInformation
from .likelihood import build_likelihood, GaussianLikelihoodInput

# --------------------------------------------------------------------------------------
# Jwst Likelihood
# --------------------------------------------------------------------------------------


@dataclass
class TargetLikelihoodSideEffects:
    plotting: ResidualPlottingInformation


def build_target_likelihood(
    response: TargetResponseInput,
    side_effect: TargetLikelihoodSideEffects,
):
    jwst_target_response: JwstResponse = build_target_response(input_config=response)

    likelihood_target = build_likelihood(
        GaussianLikelihoodInput(
            response=jwst_target_response, data=response.target_data
        )
    )

    side_effect.plotting.append_information(
        filter=response.filter,
        data=response.target_data.data,
        std=response.target_data.std,
        mask=response.target_data.mask,
        model=jwst_target_response,
    )

    return likelihood_target
