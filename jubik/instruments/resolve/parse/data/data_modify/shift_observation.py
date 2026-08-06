from configparser import ConfigParser
from dataclasses import dataclass
from typing import Union

from astropy import units as u


@dataclass
class ShiftObservation:
    shift: u.Quantity

    @classmethod
    def from_config_parser(
        cls, shift_cfg: ConfigParser | None
    ) -> Union[None, "ShiftObservation"]:
        """
        Build from ConfigParser.

        Parameters
        ----------
        shift_cfg: ConfigParser
        """
        raise NotImplementedError("Sorry not implemented. Think about it")

    @classmethod
    def from_yaml_dict(cls, shift_cfg: dict | None) -> Union[None, "ShiftObservation"]:
        """
        Build from yaml dictionary.

        Parameters
        ----------
        shift_cfg:
            - data_templates: list[tuple[float, float]]
                The shift for each data_template.
            - unit: u.Unit
        """
        if shift_cfg is None:
            return None

        shift = shift_cfg["data_templates"] * u.Unit(shift_cfg["unit"])
        return cls(shift=shift)

    def __call__(self, data_number: int) -> u.Quantity:
        return self.shift[data_number]
