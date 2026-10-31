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


class FileShifts:
    """The shift of every data file.

    A file in a group is shifted by the shared offset, domain key `shift_offset`,
    plus `shift_<group>`, a file in no group by the offset alone. Only groups
    with a file get a latent.

    Parameters
    ----------
    config: PhaseShiftConfig | None
        The parsed `shift` entry of `radio_response`; None means no shifts.
    files: Iterable[DataFile]
        Every data file of the fit, of every data block. A file loaded by
        several blocks can appear once per block.

    Raises
    ------
    ValueError
        If a file is in two groups.
    """

    def __init__(self, config: PhaseShiftConfig | None, files: Iterable[DataFile]):
        groups = () if config is None else config.groups
        group_of_file = _assign_shift_groups(groups, files)
        shifts = {}
        if config is not None:
            off = build_shift_model(OFFSET_KEY, config.offset)
            used = set(group_of_file.values())
            shifts = {None: off} | {
                g.name: _sum(off, build_shift_model(g.name, g.prior))
                for g in groups
                if g.name in used
            }
        self._shifts = {
            path: shifts.get(group) for path, group in group_of_file.items()
        }

    def of(self, file: DataFile) -> jft.Model | None:
        """The shift of one data file.

        Parameters
        ----------
        file: DataFile
            A file of `files`.

        Returns
        -------
        jft.Model | None
            The shift of `file` in arcsec, shape `(2,)`, None without `config`.

        Raises
        ------
        KeyError
            If `file` is not in `files`.
        """
        return self._shifts[file.path]


def _sum(a: jft.Model, b: jft.Model) -> jft.Model:
    return jft.Model(lambda x: a(x) + b(x), domain=a.domain | b.domain)


def _assign_shift_groups(
    groups: Iterable[ShiftGroup], files: Iterable[DataFile]
) -> dict[str, str | None]:
    """Name of the shift group of every data file by `DataFile.path`, None for
    a file in no group. Raises a ValueError if a file is in two groups."""
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
