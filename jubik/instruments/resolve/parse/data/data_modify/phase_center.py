"""Configuration for phase-center shifts."""

from dataclasses import dataclass
from typing import Union

from astropy import units as u

from ..data_loading import DataFile


@dataclass
class ShiftObservation:
    """The phase-center shift of one observation.

    Parameters
    ----------
    shift: u.Quantity
        The shift, shape `(2,)`.
    """

    shift: u.Quantity


@dataclass
class ShiftObservationConfig:
    """The `shift` entry of a data block: one shift per data template.

    Parameters
    ----------
    shifts: u.Quantity
        Shape `(n_templates, 2)`, in the order of `data_templates`.
    """

    shifts: u.Quantity

    @classmethod
    def from_yaml_dict(
        cls, shift_cfg: dict | None, n_templates: int
    ) -> Union[None, "ShiftObservationConfig"]:
        """
        Build from yaml dictionary.

        Parameters
        ----------
        shift_cfg:
            - data_templates: list[tuple[float, float]]
                The shift for each data_template, in its order.
            - unit: u.Unit
        n_templates:
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

    def __call__(self, file: DataFile) -> ShiftObservation:
        """The shift of one data file, the shift of its data template."""
        return ShiftObservation(self.shifts[file.template_index])
