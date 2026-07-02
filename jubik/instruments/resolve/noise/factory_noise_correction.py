from ..parse.noise import BaseLineCorrection, LowerBoundCorrection
from ..data.observation import Observation
from .inverse_noise_correction import InverseStandardDeviation
from .antenna_based_correction import (
    get_baselines,
    build_interferometric_noise_correction,
)
from .lower_bound_noise_correction import build_lower_bound_noise_correction

from typing import Union


def factory_noise_correction_model(
    correction_settings: Union[LowerBoundCorrection, BaseLineCorrection] | None,
    observation: Observation,
    prefix: str,
) -> InverseStandardDeviation | None:
    if correction_settings is None:
        return None

    if isinstance(correction_settings, LowerBoundCorrection):
        return build_lower_bound_noise_correction(
            alpha=correction_settings.alpha,
            scale=correction_settings.scale,
            weight=observation.weight.asnumpy(),
        )

    elif isinstance(correction_settings, BaseLineCorrection):
        return build_interferometric_noise_correction(
            *get_baselines(observation),
            alpha_correction=correction_settings.alpha,
            scale_correction=correction_settings.scale,
            weight=observation.weight.asnumpy(),
            prefix=prefix,
        )

    else:
        raise ValueError(
            "Need to pass either `LowerBoundCorrection` or `BaseLineCorrection`"
        )
