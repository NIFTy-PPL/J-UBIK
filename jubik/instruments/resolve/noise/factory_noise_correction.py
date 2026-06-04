from ..parse.noise import BaseLineCorrection, LowerBoundCorrection
from ..data.observation import Observation
from .log_inverse_noise_correction import LogInverseNoiseCovariance
from .antenna_based_correction import (
    get_baselines,
    build_antenna_based_noise_correction,
)
from .lower_bound_noise_correction import build_lower_bound_noise_correction

import nifty.cl as ift

from functools import partial
from typing import Union


def _nifty_legacy_correction(
    model: LogInverseNoiseCovariance, observation: Observation
) -> ift.JaxOperator:
    return ift.JaxOperator(
        domain=ift.makeDomain(
            {
                noise_correction_key: ift.UnstructuredDomain(val.shape)
                for noise_correction_key, val in model.domain.items()
            }
        ),
        target=observation.vis.domain,
        func=model.__call__,
    )


def factory_noise_correction_model(
    correction_settings: Union[LowerBoundCorrection, BaseLineCorrection] | None,
    observation: Observation,
    build_nifty_legacy: bool = False,
) -> Union[None, LogInverseNoiseCovariance, ift.JaxOperator]:
    if correction_settings is None:
        return None

    if build_nifty_legacy:
        wrap = partial(_nifty_legacy_correction, observation=observation)
    else:

        def wrap(x):
            return x

    if isinstance(correction_settings, LowerBoundCorrection):
        return wrap(
            build_lower_bound_noise_correction(
                alpha=correction_settings.alpha,
                scale=correction_settings.sigma,
                weight=observation.weight.val,
            )
        )

    elif isinstance(correction_settings, BaseLineCorrection):
        return wrap(
            build_antenna_based_noise_correction(
                *get_baselines(observation),
                alpha=correction_settings.alpha,
                scale=correction_settings.sigma,
                weight=observation.weight.val,
            )
        )

    else:
        raise ValueError(
            "Need to pass either `LowerBoundCorrection` or `BaseLineCorrection`"
        )
