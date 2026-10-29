"""Learnable phase shift of the model visibilities, one shift per shift group."""

import jax.numpy as jnp
import nifty.re as jft
import numpy as np
from astropy import units as u
from astropy.constants import c as speedoflight

from ...models.distributions import build_parametric_prior_from_prior_config
from ...parse.models.distributions.distributions import prior_config_factory
from .data.observation import Observation
from .parse.phase_shift import ShiftGroup
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
        The shift of the group of `observation`, shape `(2,)`.
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
