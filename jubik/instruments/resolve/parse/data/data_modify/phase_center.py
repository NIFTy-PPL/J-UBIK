"""Configuration for phase-center shifts."""

from dataclasses import dataclass
from typing import Union

from astropy import units as u

from ..data_loading import DataFile


@dataclass
class ShiftObservation:
    """The phase-center shift of one observation, see `shift_phase_center`.

    Parameters
    ----------
    shift: u.Quantity
        The sky offset `(sx, sy)` moved onto the phase center, shape `(2,)`.
    """

    shift: u.Quantity


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

    def __call__(self, file: DataFile) -> ShiftObservation:
        """The shift of a data file, the shift of its data template.

        Parameters
        ----------
        file: DataFile
            A file of the block, from `DataLoading.data_files`.
        """
        return ShiftObservation(self.shifts[file.template_index])
