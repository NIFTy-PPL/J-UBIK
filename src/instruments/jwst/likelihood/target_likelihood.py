from dataclasses import dataclass

import numpy as np

from ..data.loader.target_loader import TargetData
from ..jwst_response import (
    JwstResponse,
    TargetResponseEssentials,
    TargetResponseOptionals,
    build_target_response,
)
from ..plotting.residuals import ResidualPlottingInformation
from .likelihood import build_likelihood

# --------------------------------------------------------------------------------------
# Jwst Likelihood
# --------------------------------------------------------------------------------------


@dataclass
class TargetGaussianLikelihoodEssentials:
    """Essential input data of `load_data`."""

    response: JwstResponse
    data: TargetData


@dataclass
class TargetLikelihoodSideEffects:
    plotting: ResidualPlottingInformation


def build_target_likelihood(
    essentials: TargetResponseEssentials,
    optionals: TargetResponseOptionals,
    side_effect: TargetLikelihoodSideEffects,
):
    jwst_target_response = build_target_response(
        essentials=essentials,
        optionals=optionals,
    )

    likelihood_target = build_likelihood(
        TargetGaussianLikelihoodEssentials(
            response=jwst_target_response, data=essentials.target_data
        )
    )

    side_effect.plotting.append_information(
        filter=essentials.filter,
        data=essentials.target_data.data,
        std=essentials.target_data.std,
        mask=essentials.target_data.mask,
        model=jwst_target_response,
    )

    return likelihood_target
