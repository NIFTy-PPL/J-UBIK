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

from dataclasses import dataclass

from ....parse.models.distributions.distributions import (
    DISTRIBUTION_KEY,
    ProbabilityConfig,
    prior_config_factory,
)

PRIOR_KEY = "prior"
DATA_KEY = "data"
OFFSET_KEY = "offset"
SHIFT_SHAPE = (2,)  # sky grid (row, column)


@dataclass(frozen=True)
class ShiftGroup:
    """One learnable shift, shared by the data files it names.

    Parameters
    ----------
    name: str
        The name of the group, the latent of its shift is `shift_<name>`.
    prior: ProbabilityConfig
        The prior of the shift in arcsec, shape `(2,)`, sky grid (row, column).
    data: tuple[str, ...]
        The file names or data templates of the group, relative to the
        `data_path` of their data block.
    """

    name: str
    prior: ProbabilityConfig
    data: tuple[str, ...]

    @classmethod
    def from_yaml_dict(cls, name: str, raw: dict) -> "ShiftGroup":
        """Parse a group from its entry of `radio_response.shift`.

        Parameters
        ----------
        name: str
            The key of the group.
        raw: dict
            The group, with `prior` and `data`.

        Raises
        ------
        ValueError
            If `data` is not a non-empty list of strings.
        """
        data = raw[DATA_KEY]
        # a bare string would split into characters and drop the group's shift
        if not (
            isinstance(data, list) and data and all(isinstance(e, str) for e in data)
        ):
            raise ValueError(
                f"Shift group {name}: `{DATA_KEY}` must be a non-empty list of "
                f"file names or templates, got {data!r}"
            )
        return cls(
            name=name,
            prior=prior_config_factory(raw[PRIOR_KEY], shape=SHIFT_SHAPE),
            data=tuple(data),
        )


@dataclass(frozen=True)
class PhaseShiftConfig:
    """The `shift` entry of `radio_response`.

    Parameters
    ----------
    offset: ProbabilityConfig
        The prior of the offset in arcsec, shape `(2,)`, sky grid (row, column).
    groups: tuple[ShiftGroup, ...]
        The optional shift groups.
    """

    offset: ProbabilityConfig
    groups: tuple[ShiftGroup, ...]

    @classmethod
    def from_yaml_dict(cls, raw: dict | None) -> "PhaseShiftConfig | None":
        """Parse the `shift` entry of `radio_response`.

        Parameters
        ----------
        raw: dict | None
            The `shift` entry; None means no shifts.

        Returns
        -------
        PhaseShiftConfig | None
            None if `raw` is None.

        Raises
        ------
        ValueError
            If `offset` is missing or written as a group, or a group is invalid,
            see `ShiftGroup.from_yaml_dict`.
        """
        if raw is None:
            return None
        offset = raw.get(OFFSET_KEY)
        if offset is None:
            raise ValueError(
                f"`radio_response.shift` needs an `{OFFSET_KEY}` prior: it moves "
                "every file and the groups add to it; leave `shift` out for no shifts"
            )
        # a prior dict names its distribution; `prior` or `data` keys mean a group
        if isinstance(offset, dict) and DISTRIBUTION_KEY not in offset:
            raise ValueError(
                f"Shift `{OFFSET_KEY}` is the prior itself, e.g. `{OFFSET_KEY}: "
                f"[normal, 0.0, 0.2]`, not a group; the name is reserved, give a "
                f"group another name, got {offset!r}"
            )
        return cls(
            offset=prior_config_factory(offset, shape=SHIFT_SHAPE),
            groups=tuple(
                ShiftGroup.from_yaml_dict(name, group)
                for name, group in raw.items()
                if name != OFFSET_KEY
            ),
        )
