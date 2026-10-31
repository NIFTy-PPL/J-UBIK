"""Learnable phase shift of the model visibilities: an offset shared by every
file, plus one shift per optional shift group."""

from collections import defaultdict
from collections.abc import Iterable

import jax.numpy as jnp
import nifty.re as jft
import numpy as np
from astropy import units as u
from astropy.constants import c as speedoflight
from nifty.cl.logger import logger

from ...models.distributions import build_parametric_prior_from_prior_config
from ...parse.models.distributions.distributions import ProbabilityConfig
from .data.observation import Observation
from .parse.data.data_loading import DataFile
from .parse.phase_shift import OFFSET_KEY, SHIFT_SHAPE, PhaseShiftConfig, ShiftGroup
from .util import cast_to_dtype


def build_shift_model(name: str, prior: ProbabilityConfig) -> jft.Model:
    """Prior of one shift in arcsec.

    Parameters
    ----------
    name: str
        The name of the shift, the domain key is `shift_<name>`.
    prior: ProbabilityConfig
        The prior, shape `(2,)`, sky grid (row, column).

    Returns
    -------
    jft.Model
        The shift, shape `(2,)`.
    """
    return build_parametric_prior_from_prior_config(
        f"shift_{name}", prior, SHIFT_SHAPE, as_model=True
    )


class PhaseShift(jft.Model):
    """Phase factor `exp(1j * (u * shift_col - v * shift_row))` of the model
    visibilities, cast to `dtype`, see `build_phase_shift`.

    Parameters
    ----------
    shift: jft.Model
        The shift in arcsec, shape `(2,)`, sky grid (row, column).
    u: np.ndarray
        Phase in rad per arcsec of column shift, shape `(1, n_rows, n_freq)`.
    v: np.ndarray
        Phase in rad per arcsec of row shift, shape `(1, n_rows, n_freq)`.
    dtype: np.dtype
        The dtype of the visibilities.
    """

    def __init__(self, shift: jft.Model, u: np.ndarray, v: np.ndarray, dtype):
        self.shift = shift
        self._u = u
        self._v = v
        self._dtype = dtype
        super().__init__(domain=shift.domain)

    def __call__(self, x):
        shift_row, shift_col = self.shift(x)
        phase = jnp.exp(1j * (self._u * shift_col - self._v * shift_row))
        return cast_to_dtype(phase, self._dtype)


def build_phase_shift(
    config: PhaseShiftConfig | None, group: ShiftGroup | None, observation: Observation
) -> PhaseShift | None:
    """Phase factor of the model visibilities of one observation for its shift.

    The shift is the offset, domain key `shift_offset`, plus `shift_<group>` if
    the observation's file is in `group`. Observations of one group share the
    latent through its domain key.

    The shift moves the sampling positions of the model sky by `shift` along
    the sky grid axes (row, column), in arcsec, so the sky as the data see it
    moves by `-shift`, as the JWST shift does. With (u, v) in wavelengths and
    the shift in radians the factor is `exp(2j pi (u * shift_col - v * shift_row))`,
    pinned by `test/instruments/resolve/test_phase_shift.py`. The w term is
    ignored (narrow field).

    Parameters
    ----------
    config: PhaseShiftConfig | None
        The parsed `shift` entry of `radio_response`; None means no shifts.
    group: ShiftGroup | None
        The group of the observation's file, see `assign_shift_groups`; None
        for a file in no group.
    observation: Observation
        Provides `uvw`, `freq` and the dtype of the visibilities.

    Returns
    -------
    PhaseShift | None
        The phase factor, shape `(1, n_rows, n_freq)`, None without `config`.
    """
    if config is None:
        return None
    shift = build_shift_model(OFFSET_KEY, config.offset)
    if group is not None:
        off, grp = shift, build_shift_model(group.name, group.prior)
        shift = jft.Model(lambda x: off(x) + grp(x), domain=off.domain | grp.domain)

    scale = 2 * np.pi * u.arcsec.to(u.rad) / speedoflight.value
    uv = observation.uvw[:, :2, None] * observation.freq * scale
    # (1, n_rows, n_freq) broadcasts against the visibilities (n_pol, n_rows, n_freq)
    return PhaseShift(shift, uv[None, :, 0], uv[None, :, 1], observation.vis_val.dtype)


def assign_shift_groups(
    config: PhaseShiftConfig | None, files: Iterable[DataFile]
) -> dict[str, ShiftGroup | None]:
    """The shift group of every data file of the fit.

    Parameters
    ----------
    config: PhaseShiftConfig | None
        The parsed `shift` entry of `radio_response`; None means no shifts.
    files: Iterable[DataFile]
        Every data file of the fit, of every data block. A file loaded by
        several blocks can appear once per block and gets the group named
        through any of them.

    Returns
    -------
    dict[str, ShiftGroup | None]
        The group of each file by `DataFile.path`, None for a file in no group.

    Raises
    ------
    ValueError
        If a file is in two groups. Entries that match no file are ignored with
        a warning.
    """
    groups = {} if config is None else {g.name: g for g in config.groups}
    used = set()
    owner = defaultdict(set)  # groups of each file, across blocks
    for file in files:
        hits = {(g.name, e) for g in groups.values() for e in g.data if file.matches(e)}
        owner[file.path] |= {name for name, _ in hits}
        used |= hits

    for path, names in owner.items():
        if len(names) > 1:
            raise ValueError(f"{path} is in the shift groups {sorted(names)}")

    unused = [
        (g.name, e) for g in groups.values() for e in g.data if (g.name, e) not in used
    ]
    if unused:
        logger.warning(f"Shift entries match no loaded data file, ignored: {unused}")
    return {
        path: groups[next(iter(names))] if names else None
        for path, names in owner.items()
    }
