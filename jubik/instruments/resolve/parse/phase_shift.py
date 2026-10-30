"""Parse `radio_response.shift`: learnable shifts and the data files sharing them.

```yaml
radio_response:
  shift:
    offset:
      prior: [normal, 0.0, 0.2]    # arcsec, shared by every file
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
block. Files of several data blocks can share a shift. An entry that matches no
file of the loaded blocks, e.g. of a block left out of the fit, is ignored with
a warning.

The optional `offset` is not a group: it has only a `prior`, latent
`shift_offset`, and moves every loaded file. A file in a group is shifted by
`offset + shift_<group>`, a file in no group by `offset` alone, or not at all
without `offset`. The name is reserved, so no group can be called `offset`.
"""

from collections import defaultdict
from dataclasses import dataclass
from os.path import join, normpath

from nifty.cl.logger import logger

from .data.data_loading import DataLoading

PRIOR_KEY = "prior"
DATA_KEY = "data"
OFFSET_KEY = "offset"


@dataclass
class ShiftGroup:
    """One learnable shift: its prior and the data files that share it."""

    name: str
    prior: list
    data: list[str]


def parse_shift_groups(shift_cfg: dict | None) -> list[ShiftGroup]:
    """Read the groups of the `shift` entry of `radio_response`; None: no shifts."""
    if shift_cfg is None:
        return []
    groups = []
    for name, group in shift_cfg.items():
        if name == OFFSET_KEY:
            continue
        data = group[DATA_KEY]
        # a bare string would split into characters and drop the group's shift
        if not (
            isinstance(data, list) and data and all(isinstance(e, str) for e in data)
        ):
            raise ValueError(
                f"Shift group {name}: `{DATA_KEY}` must be a non-empty list of "
                f"file names or templates, got {data!r}"
            )
        groups.append(ShiftGroup(name=name, prior=group[PRIOR_KEY], data=data))
    return groups


def parse_shift_offset(shift_cfg: dict | None) -> list | None:
    """Prior of the `offset` of the `shift` entry, None without one."""
    offset = (shift_cfg or {}).get(OFFSET_KEY)
    if offset is None:
        return None
    if DATA_KEY in offset:
        raise ValueError(
            f"Shift `{OFFSET_KEY}` moves every file and takes only a `{PRIOR_KEY}`; "
            f"the name is reserved, give the group of {offset[DATA_KEY]!r} another name"
        )
    return offset[PRIOR_KEY]


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
        `DataLoading.loader_templates`, None for a file in no group. A file
        loaded by several blocks has its group in all of them.

    Raises
    ------
    ValueError
        If a file is in two groups.
    """
    used = set()
    owner = defaultdict(set)  # groups of each file, across blocks
    files = {}
    for block, data_cfg in data_cfgs.items():
        data_path = data_cfg["data_path"]
        files[block] = []
        for lt in DataLoading.from_yaml_dict(data_cfg).loader_templates:
            file = normpath(lt.file_path)
            matches = {normpath(lt.data_template), file}
            hits = {
                (g.name, entry)
                for g in groups
                for entry in g.data
                if normpath(join(data_path, entry)) in matches
            }
            owner[file] |= {name for name, _ in hits}
            used |= hits
            files[block].append(file)

    for file, names in owner.items():
        if len(names) > 1:
            raise ValueError(f"{file} is in the shift groups {sorted(names)}")

    unused = [(g.name, e) for g in groups for e in g.data if (g.name, e) not in used]
    if unused:
        logger.warning(f"Shift entries match no loaded data file, ignored: {unused}")
    return {
        block: [next(iter(owner[file]), None) for file in block_files]
        for block, block_files in files.items()
    }
