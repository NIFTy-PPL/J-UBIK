"""Parse `radio_response.shift`: learnable shifts and the data files sharing them.

```yaml
radio_response:
  shift:
    b4_eb1:
      prior: [normal, 0.0, 0.05]   # arcsec, sky grid (row, column)
      data:
        - uid_A/uid_A_spw{spw}.npz   # template: every spw of the EB
    b4_eb1_usb:
      prior: [normal, 0.0, 0.1]
      data:
        - uid_B/uid_B_spw17.npz      # expanded file: this spw only
```

A `data` entry is a data template exactly as written in `data_templates`, or
one file the template expands to, both relative to the `data_path` of the data
block. Files of several data blocks can share a shift. A file in no group has
no shift. An entry that matches no file of the loaded blocks, e.g. of a block
left out of the fit, is ignored with a warning.
"""

from dataclasses import dataclass
from os.path import join, normpath

from nifty.cl.logger import logger

from .data.data_loading import DataLoading

PRIOR_KEY = "prior"
DATA_KEY = "data"


@dataclass
class ShiftGroup:
    """One learnable shift: its prior and the data files that share it."""

    name: str
    prior: list
    data: list[str]


def parse_shift_groups(shift_cfg: dict | None) -> list[ShiftGroup]:
    """Read the `shift` entry of `radio_response`; None means no shifts."""
    if shift_cfg is None:
        return []
    return [
        ShiftGroup(name=name, prior=group[PRIOR_KEY], data=list(group[DATA_KEY]))
        for name, group in shift_cfg.items()
    ]


def assign_shift_groups(
    groups: list[ShiftGroup], data_cfgs: dict[str, dict]
) -> dict[str, list[str | None]]:
    """Name of the shift group of every file of every data block.

    Parameters
    ----------
    groups: list[ShiftGroup]
        The shift groups.
    data_cfgs: dict[str, dict]
        The config of each data block, by data block name.

    Returns
    -------
    dict[str, list[str | None]]
        Per data block, the group of each file in the order of
        `DataLoading.loader_templates`, None for a file in no group.

    Raises
    ------
    ValueError
        If a file is in two groups.
    """
    used = set()
    assigned = {}
    for block, data_cfg in data_cfgs.items():
        data_path = data_cfg["data_path"]
        names = []
        for lt in DataLoading.from_yaml_dict(data_cfg).loader_templates:
            files = {normpath(lt.data_template), normpath(lt.file_path)}
            hits = {
                (g.name, entry)
                for g in groups
                for entry in g.data
                if normpath(join(data_path, entry)) in files
            }
            hit_names = sorted({name for name, _ in hits})
            if len(hit_names) > 1:
                raise ValueError(f"{lt.file_path} is in the shift groups {hit_names}")
            used |= hits
            names.append(hit_names[0] if hit_names else None)
        assigned[block] = names

    unused = [(g.name, e) for g in groups for e in g.data if (g.name, e) not in used]
    if unused:
        logger.warning(f"Shift entries match no loaded data file, ignored: {unused}")
    return assigned
