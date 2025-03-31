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

    @classmethod
    def from_yaml_dict(cls, settings: dict):
        return cls(mode=settings.get("mode", "constant"))


@dataclass
class SparseConfig:
    extend_factor: int = 1
    to_bottom_left: bool = False

    @classmethod
    def from_yaml_dict(cls, settings: dict):
        return cls(
            extend_factor=settings.get("extend_factor", 1),
            to_bottom_left=settings.get("to_bottom_left", False),
        )


# Factory function implementing the Strategy Pattern
def rotation_and_shift_algorithm_config_factory(
    config_dict: dict[str, Any],
) -> Union[LinearConfig, NufftConfig, SparseConfig]:
    strategy_map = {
        "linear": LinearConfig,
        "nufft": NufftConfig,
        "sparse": SparseConfig,
    }

    for key, config_class in strategy_map.items():
        if key in config_dict and config_dict[key] is not None:
            return config_class.from_yaml_dict(config_dict[key])

    raise ValueError(
        "No matching `rotation_and_shift` algorithm found.\n"
        f"Options: {strategy_map.keys()}"
    )
