from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp

from ...parsing_base import FromYamlDict


@dataclass
class SimpleSpectralSkyConfig(FromYamlDict):
    reference_bin: int
    zero_mode: tuple | list | Callable
    spatial_amplitude: dict
    spectral_index: dict
    spectral_amplitude: dict | None = None
    spectral_deviations: dict | None = None
    spatial_amplitude_model: str = "non_parametric"
    spectral_amplitude_model: str = "non_parametric"
    harmonic_type: str = "fourier"
    nonlinearity: Callable = jnp.exp

    @classmethod
    def from_yaml_dict(cls, raw: dict) -> "SimpleSpectralSkyConfig":
        return cls(
            reference_bin=raw["reference_bin"],
            zero_mode=raw["zero_mode"],
            spatial_amplitude=raw["spatial_amplitude"],
            spectral_index=raw["spectral_index"],
            spectral_amplitude=raw.get("spectral_amplitude"),
            spectral_deviations=raw.get("spectral_deviations"),
            spatial_amplitude_model=raw["spatial_amplitude"].get(
                "model", "non_parametric"
            ),
            spectral_amplitude_model=raw.get("spectral_amplitude", {}).get(
                "model", "non_parametric"
            ),
            harmonic_type=raw.get("harmonic_type", "fourier"),
            nonlinearity=getattr(jnp, raw.get("nonlinearity", "exp")),
        )
