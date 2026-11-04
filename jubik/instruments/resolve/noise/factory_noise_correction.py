from ..parse.noise import NoiseCorrection
from ..data.observation import Observation
from .inverse_noise_correction import InverseStandardDeviation
from .antenna_based_correction import build_interferometric_noise_correction


def factory_noise_correction_model(
    correction_settings: NoiseCorrection | None,
    observation: Observation,
    prefix: str,
) -> InverseStandardDeviation | None:
    if correction_settings is None:
        return None

    if isinstance(correction_settings, NoiseCorrection):
        return build_interferometric_noise_correction(
            observation, correction_settings, prefix=prefix
        )

    else:
        raise ValueError("Need to pass `NoiseCorrection`")
