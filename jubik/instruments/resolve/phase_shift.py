"""Learnable phase shift of the model visibilities, one shift per observation."""

import jax.numpy as jnp
import nifty.re as jft
import numpy as np
from astropy import units as u
from astropy.constants import c as speedoflight

from ...models.distributions import build_parametric_prior_from_prior_config
from ...parse.models.distributions.distributions import prior_config_factory
from .data.observation import Observation
from .util import cast_to_dtype

DEFAULT_KEY = "default"


def build_shift_prior(
    data_name: str, shift_settings: dict | list | None, n_observations: int
) -> jft.Model | None:
    """Prior of the shifts of all observations of the band `data_name`, in arcsec.

    Parameters
    ----------
    data_name: str
        The band; the model has the domain key `<data_name>_shift`.
    shift_settings: dict | list | None
        The `shift` entry of `radio_response`. Either a prior
        `[distribution, mean, std]` for every band, or a dict holding a
        `default` prior and optionally priors under band names, which win over
        `default`. None means no shift.
    n_observations: int
        Number of observations of the band; row k belongs to observation k.

    Returns
    -------
    jft.Model | None
        The shifts, shape `(n_observations, 2)`, (row, column) of the sky grid.
    """
    if shift_settings is None:
        return None
    if isinstance(shift_settings, dict):
        key = data_name if data_name in shift_settings else DEFAULT_KEY
        shift_settings = shift_settings[key]

    shape = (n_observations, 2)
    return build_parametric_prior_from_prior_config(
        f"{data_name}_shift",
        prior_config_factory(shift_settings, shape=shape),
        shape,
        as_model=True,
    )


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
        The shifts of all observations of a band, shape `(n_observations, 2)`.
    index: int
        The row of `shift` that belongs to `observation`.
    observation: Observation
        Provides `uvw`, `freq` and the dtype of the visibilities.
    """

    def __init__(self, shift: jft.Model, index: int, observation: Observation):
        self.shift = shift
        self.index = index
        scale = 2 * np.pi * u.arcsec.to(u.rad) / speedoflight.value
        uv = observation.uvw[:, :2, None] * observation.freq * scale
        # (1, n_rows, n_freq) broadcasts against the visibilities (n_pol, n_rows, n_freq)
        self._u = uv[None, :, 0]
        self._v = uv[None, :, 1]
        self._dtype = observation.vis_val.dtype
        super().__init__(domain=shift.domain)

    def __call__(self, x):
        shift_row, shift_col = self.shift(x)[self.index]
        phase = jnp.exp(1j * (self._u * shift_col - self._v * shift_row))
        return cast_to_dtype(phase, self._dtype)
