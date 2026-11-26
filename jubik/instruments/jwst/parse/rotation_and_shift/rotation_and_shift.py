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
    # cycles per arcsec; None keeps every sky mode; "auto" (the config's default) is
    # resolved per filter by the likelihood builder
    band_limit: float | str | None = None

    @classmethod
    def from_yaml_dict(cls, settings: dict):
        band = settings.get("band_limit", "auto")
        if band is None or band is False or band == "off":
            band = None
        elif band != "auto":
            if (
                isinstance(band, bool)
                or not isinstance(band, (int, float))
                or band <= 0
            ):
                raise ValueError(
                    "rotation_and_shift.nufft.band_limit: a number > 0 (cycles per "
                    f"arcsec), 'auto' or 'off', got {band!r}"
                )
            band = float(band)
        return cls(mode=settings.get("mode", "constant"), band_limit=band)


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
