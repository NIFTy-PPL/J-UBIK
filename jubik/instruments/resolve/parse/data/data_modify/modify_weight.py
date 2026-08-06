from configparser import ConfigParser
from dataclasses import dataclass
from typing import Union


@dataclass
class SystematicErrorBudget:
    percentage: float

    @classmethod
    def from_config_parser(
        cls, data_cfg: ConfigParser
    ) -> Union[None, "SystematicErrorBudget"]:
        """
        Build from ConfigParser.

        Parameters
        ----------
        systematic error budget:
            The percentage of the absolute value of the visibilities to be
            added to the sigma (weight) of the visibilities (1-5)% is adviced.
        """
        percentage = eval(data_cfg.get("systematic error budget", "None"))

        return (
            None
            if percentage is None
            else SystematicErrorBudget(percentage=float(percentage))
        )

    @classmethod
    def from_yaml_dict(
        cls, systematic_error: dict
    ) -> Union[None, "SystematicErrorBudget"]:
        """
        Build from Yaml dictionary.

        Parameters
        ----------
        percentage:
            The percentage of the absolute value of the visibilities to be
            added to the sigma (weight) of the visibilities (1-5)% is adviced.
        """
        percentage = systematic_error.get("percentage")
        return (
            None
            if percentage is None
            else SystematicErrorBudget(percentage=float(percentage))
        )
