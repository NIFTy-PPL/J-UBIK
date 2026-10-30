"""Learnable phase shift of the model visibilities: an offset shared by every
file, plus one shift per optional shift group."""

import jax.numpy as jnp
import nifty.re as jft
import numpy as np
from astropy import units as u
from astropy.constants import c as speedoflight

from ...models.distributions import build_parametric_prior_from_prior_config
from ...parse.models.distributions.distributions import prior_config_factory
from .data.observation import Observation
from .parse.phase_shift import (
    OFFSET_KEY,
    ShiftGroup,
    assign_shift_groups,
    parse_shift_groups,
    parse_shift_offset,
)
from .util import cast_to_dtype


def build_shift_models(groups: list[ShiftGroup]) -> dict[str, jft.Model]:
    """Prior of the shift of each group, in arcsec, by group name.

    The model of group `name` has the domain key `shift_<name>` and shape `(2,)`,
    (row, column) of the sky grid.
    """
    return {
        g.name: build_parametric_prior_from_prior_config(
            f"shift_{g.name}",
            prior_config_factory(g.prior, shape=(2,)),
            (2,),
            as_model=True,
        )
        for g in groups
    }


def build_file_shifts(
    shift_cfg: dict | None, data_cfgs: dict[str, dict]
) -> dict[str, list[jft.Model | None]]:
    """The shift of every file of every data block, None for a file without one.

    A file in a group is shifted by the shared offset, domain key `shift_offset`,
    plus `shift_<group>`; a file in no group by the offset alone; no file
    without `shift_cfg`. `shift_cfg` is the `shift` entry of `radio_response`,
    `data_cfgs` the config of each data block by name; the files are in the
    order of `assign_shift_groups`.
    """
    shift_groups = parse_shift_groups(shift_cfg)
    group_of_file = assign_shift_groups(shift_groups, data_cfgs)
    used = {group for groups in group_of_file.values() for group in groups}
    shifts = build_shift_models([g for g in shift_groups if g.name in used])
    offset = parse_shift_offset(shift_cfg)
    if offset is not None:
        off = build_shift_models([ShiftGroup(OFFSET_KEY, offset, [])])[OFFSET_KEY]
        shifts = {None: off} | {
            name: jft.Model(lambda x, s=s: off(x) + s(x), domain=off.domain | s.domain)
            for name, s in shifts.items()
        }
    return {
        block: [shifts.get(group) for group in groups]
        for block, groups in group_of_file.items()
    }


class PhaseShift(jft.Model):
    """Phase factor of the model visibilities of one observation for its shift.

    The shift moves the sampling positions of the model sky by `shift` along
    the sky grid axes (row, column), in arcsec, so the sky as the data see it
    moves by `-shift`, as the JWST shift does. With (u, v) in wavelengths and
    the shift in radians the factor is `exp(2j pi (u * shift_col - v * shift_row))`,
    pinned by `test/instruments/resolve/test_phase_shift.py`. The w term is
    ignored (narrow field).

    Parameters
    ----------
    shift: jft.Model
        The shift of `observation`, shape `(2,)`.
    observation: Observation
        Provides `uvw`, `freq` and the dtype of the visibilities.
    """

    def __init__(self, shift: jft.Model, observation: Observation):
        self.shift = shift
        scale = 2 * np.pi * u.arcsec.to(u.rad) / speedoflight.value
        uv = observation.uvw[:, :2, None] * observation.freq * scale
        # (1, n_rows, n_freq) broadcasts against the visibilities (n_pol, n_rows, n_freq)
        self._u = uv[None, :, 0]
        self._v = uv[None, :, 1]
        self._dtype = observation.vis_val.dtype
        super().__init__(domain=shift.domain)

    def __call__(self, x):
        shift_row, shift_col = self.shift(x)
        phase = jnp.exp(1j * (self._u * shift_col - self._v * shift_row))
        return cast_to_dtype(phase, self._dtype)
