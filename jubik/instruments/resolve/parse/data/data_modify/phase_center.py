"""Configuration for phase-center shifts."""

from dataclasses import dataclass

from astropy import units as u


@dataclass
class ShiftObservation:
    shift: u.Quantity


def shifts_by_template(
    data_cfg: dict,
) -> tuple[tuple[str, ShiftObservation], ...] | None:
    """
    Build the shift of each data template from the yaml dictionary of a data
    block.

    Parameters
    ----------
    data_cfg:
        The data block, with `data_templates` and
        - shift: dict | None
            - data_templates: list[tuple[float, float]]
                The shift for each data_template, in its order.
            - unit: u.Unit

    Returns
    -------
    tuple[tuple[str, ShiftObservation], ...] | None
        Each data template, as written in `data_templates`, with its shift;
        None without `shift`.

    Raises
    ------
    ValueError
        If there is not one shift per data template.
    """
    shift_cfg = data_cfg.get("shift")
    if shift_cfg is None:
        return None

    templates = data_cfg["data_templates"]
    shifts = shift_cfg["data_templates"] * u.Unit(shift_cfg["unit"])
    if len(shifts) != len(templates):
        raise ValueError(
            f"`shift.data_templates` needs one shift per data template, got "
            f"{len(shifts)} shifts for {len(templates)} templates"
        )
    return tuple((t, ShiftObservation(s)) for t, s in zip(templates, shifts))
