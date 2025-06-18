from dataclasses import dataclass

import nifty8.re as jft

from ..jwst_response import (
    JwstResponse,
)
from ..data.loader.target_loader import TargetData
from ..plotting.residuals import ResidualPlottingInformation
from .likelihood import GaussianLikelihoodBuilder

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
    builder: GaussianLikelihoodBuilder

    @property
    def likelihood(self) -> jft.Likelihood | jft.Gaussian:
        return self.builder.build()


# Interface ----------------------------------------------------------------------------


def build_target_likelihood(
    response: JwstResponse,
    target_data: TargetData,
    filter_name: str,
    side_effect: TargetLikelihoodSideEffects | None = None,
):
    builder = GaussianLikelihoodBuilder(
        response=response,
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
            model=response,
        )

    return SingleTargetLikelihood(filter=filter_name, builder=builder)
