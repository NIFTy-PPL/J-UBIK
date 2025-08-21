from configparser import ConfigParser
from dataclasses import dataclass

from .....parse.parsing_base import StaticTyped


@dataclass
class LowerBoundCorrection(StaticTyped):
    alpha: float | int
    sigma: float | int

    @classmethod
    def from_config_parser(cls, config: ConfigParser):
        """Build `LowerBoundCorrection` from `ConfigParser`.

        Parameters
        ----------
        config: ConfigParser
            `noise correction alpha`
            `noise correction sigma`
        """
        return LowerBoundCorrection(
            alpha=float(config["noise correction alpha"]),
            sigma=float(config["noise correction sigma"]),
        )

    @classmethod
    def from_yaml_dict(cls, config: dict):
        raise NotImplementedError
