"""Configuration for phase-center shifts."""

from dataclasses import dataclass
from typing import Union

from astropy import units as u


@dataclass
class ShiftObservationConfig:
    """The `shift` entry of a data block, one shift per data template.

    Parameters
    ----------
    shifts: u.Quantity
        The shift of each data template, in the order of `data_templates`,
        shape `(n_templates, 2)`.
    """

    shifts: u.Quantity

    @classmethod
    def from_yaml_dict(
        cls, shift_cfg: dict | None, n_templates: int
    ) -> Union[None, "ShiftObservationConfig"]:
        """Build from the `shift` entry of a data block.

        Parameters
        ----------
        shift_cfg: dict | None
            None means no shift, otherwise
            - data_templates: list[tuple[float, float]]
                The shift `(sx, sy)` of each data template, in its order.
            - unit: str
                The unit of the shifts, e.g. `arcsec`.
        n_templates: int
            The number of data templates of the data block.

        Raises
        ------
        ValueError
            If there is not one shift per data template.
        """
        if shift_cfg is None:
            return None

        shifts = shift_cfg["data_templates"] * u.Unit(shift_cfg["unit"])
        if len(shifts) != n_templates:
            raise ValueError(
                f"`shift.data_templates` needs one shift per data template, got "
                f"{len(shifts)} shifts for {n_templates} templates"
            )
        return cls(shifts=shifts)

