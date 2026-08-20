from ..parse.noise import BaseLineCorrection
from ..data.observation import Observation
from .inverse_noise_correction import InverseStandardDeviation
from .antenna_based_correction import (
    get_baselines,
    build_interferometric_noise_correction,
)


def factory_noise_correction_model(
    correction_settings: BaseLineCorrection | None,
    observation: Observation,
    prefix: str,
) -> InverseStandardDeviation | None:
    if correction_settings is None:
        return None

    if isinstance(correction_settings, BaseLineCorrection):
        return build_interferometric_noise_correction(
            *get_baselines(observation),
            alpha_correction=correction_settings.alpha,
            scale_correction=correction_settings.scale,
            weight=observation.weight.asnumpy(),
            prefix=prefix,
        )

    else:
        raise ValueError("Need to pass `BaseLineCorrection`")
