from configparser import ConfigParser
from dataclasses import dataclass

from .....parse.parsing_base import StaticTyped


@dataclass
class BaseLineCorrection(StaticTyped):
    alpha: float | int
    scale: float | int

    @classmethod
    def from_yaml_dict(cls, config: dict):
        return cls(alpha=config["alpha"], scale=config["scale"])
