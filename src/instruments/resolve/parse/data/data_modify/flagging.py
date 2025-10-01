from configparser import ConfigParser
from dataclasses import dataclass

from ......parse.parsing_base import StaticTyped


@dataclass
class FlagWeights(StaticTyped):
    min: float = 1e-12
    max: float = 1e12

    @classmethod
    def from_yaml_dict(cls, raw: dict) -> "FlagWeights":
        return cls(
            min=raw.get("min", 1e-12),
            max=raw.get("max", 1e12),
        )

    @classmethod
    def from_config_parser(cls, raw: ConfigParser | dict) -> "FlagWeights":
        return cls(
            min=eval(raw.get("min", "1e-12")),
            max=eval(raw.get("max", "1e+12")),
        )
