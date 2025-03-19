from ....parsing_base import StronglyTyped

from dataclasses import dataclass
from configparser import ConfigParser


@dataclass
class BaseLineCorrection(StronglyTyped):
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
