from configparser import ConfigParser
from dataclasses import dataclass

from .....parse.parsing_base import StaticTyped


@dataclass
class BaseLineCorrection(StaticTyped):
    alpha: float | int
    sigma: float | int

    @classmethod
    def from_config_parser(cls, config: ConfigParser):
        return BaseLineCorrection(
            alpha=float(config["noise correction alpha"]),
            sigma=False,  # float(data_cfg['noise correction sigma']),
        )

    @classmethod
    def from_yaml_dict(cls, config: dict):
        raise NotImplementedError
