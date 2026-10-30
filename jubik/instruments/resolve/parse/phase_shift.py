"""Parse `radio_response.shift`: a shared offset and optional shift groups.

```yaml
radio_response:
  shift:
    offset: [normal, 0.0, 0.2]     # arcsec, sky grid (row, column), every file
    b4_eb1:
      prior: [normal, 0.0, 0.05]   # arcsec, added to the offset
      data:
        - uid_A/uid_A_spw{spw}.npz   # template: every spw of the EB
    b4_eb1_usb:
      prior: [normal, 0.0, 0.1]
      data:
        - uid_B/uid_B_spw17.npz      # expanded file: this spw only
```

With `shift` set, `offset` is required. It is not a group but the prior
itself, latent `shift_offset`, and moves every loaded file. The groups are
optional: a file in a group is shifted by `offset + shift_<group>`, a file in
no group by `offset` alone. Without `shift` no file is shifted. The name
`offset` is reserved, so no group can be called `offset`.

Reference configuration: leave at least one file in no group, i.e. keep its
data but list it in no group's `data`. If every file is in a group, `offset + a`
and `shift_<group> - a` for every group give the same visibilities for any `a`,
so only the priors split the shift between the offset and the groups. A file in
no group pins the offset to its own shift. This does not by itself anchor the
absolute position of the sky.

A `data` entry is a data template exactly as written in `data_templates`, or
one file the template expands to, both relative to the `data_path` of the data
block. Files of several data blocks can share a shift. An entry that matches no
file of the loaded blocks, e.g. of a block left out of the fit, is ignored with
a warning.
"""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from nifty.cl.logger import logger

from ....parse.models.distributions.distributions import DISTRIBUTION_KEY
from .data.data_loading import DataFile

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


def parse_shift_offset(shift_cfg: dict | None) -> list | dict | None:
    """Prior of the `offset` of the `shift` entry; None means no shifts."""
    if shift_cfg is None:
        return None
    offset = shift_cfg.get(OFFSET_KEY)
    if offset is None:
        raise ValueError(
            f"`radio_response.shift` needs an `{OFFSET_KEY}` prior: it moves every "
            "file and the groups add to it; leave `shift` out for no shifts"
        )
    # a prior dict names its distribution; `prior` or `data` keys mean a group
    if isinstance(offset, dict) and DISTRIBUTION_KEY not in offset:
        raise ValueError(
            f"Shift `{OFFSET_KEY}` is the prior itself, e.g. `{OFFSET_KEY}: "
            f"[normal, 0.0, 0.2]`, not a group; the name is reserved, give a group "
            f"another name, got {offset!r}"
        )
    return offset


def assign_shift_groups(
    groups: list[ShiftGroup], files: Iterable[DataFile]
) -> dict[str, str | None]:
    """Name of the shift group of every data file.

    Parameters
    ----------
    groups: list[ShiftGroup]
        The shift groups.
    files: Iterable[DataFile]
        Every loaded data file, of every data block. A file loaded by several
        blocks can appear once per block.

    Returns
    -------
    dict[str, str | None]
        The group of each file by `DataFile.path`, None for a file in no group.

    Raises
    ------
    ValueError
        If a file is in two groups.
    """
    used = set()
    owner = defaultdict(set)  # groups of each file, across blocks
    for file in files:
        hits = {(g.name, e) for g in groups for e in g.data if file.matches(e)}
        owner[file.path] |= {name for name, _ in hits}
        used |= hits

    for path, names in owner.items():
        if len(names) > 1:
            raise ValueError(f"{path} is in the shift groups {sorted(names)}")

    unused = [(g.name, e) for g in groups for e in g.data if (g.name, e) not in used]
    if unused:
        logger.warning(f"Shift entries match no loaded data file, ignored: {unused}")
    return {path: next(iter(names), None) for path, names in owner.items()}
