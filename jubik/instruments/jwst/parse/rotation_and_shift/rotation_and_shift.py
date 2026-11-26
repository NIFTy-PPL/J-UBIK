from enum import Enum
from dataclasses import dataclass
from typing import Any, Union


@dataclass
class LinearConfig:
    order: int = 1
    mode: str = "constant"

    @classmethod
    def from_yaml_dict(cls, settings: dict):
        return cls(
            order=settings.get("order", 1), mode=settings.get("mode", "constant")
        )


@dataclass
class NufftConfig:
    mode: str = "constant"
    band_limit: float | None = None  # cycles per arcsec; None keeps every sky mode
    # PSF, pixel and shift as transfer functions, one point per detector pixel
    # (`build_fourier_response`); needs band_limit and mode constant
    fourier: bool = False

    @classmethod
    def from_yaml_dict(cls, settings: dict):
        return cls(
            mode=settings.get("mode", "constant"),
            band_limit=settings.get("band_limit"),
            fourier=settings.get("fourier", False),
        )


# Factory function implementing the Strategy Pattern
def rotation_and_shift_algorithm_config_factory(
    config_dict: dict[str, Any],
) -> Union[LinearConfig, NufftConfig]:
    strategy_map = {
        "linear": LinearConfig,
        "nufft": NufftConfig,
    }

    for key, config_class in strategy_map.items():
        if key in config_dict and config_dict[key] is not None:
            return config_class.from_yaml_dict(config_dict[key])

    raise ValueError(
        "No matching `rotation_and_shift` algorithm found.\n"
        f"Options: {strategy_map.keys()}"
    )
