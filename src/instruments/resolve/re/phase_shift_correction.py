from typing import Union

import jax.numpy as jnp
import nifty8.re as jft
import numpy as np
from astropy.constants import c as speedoflight
from astropy import units

from ...jwst.parse.rotation_and_shift.coordinates_correction import (
    CoordinatesCorrectionPriorConfig,
)
from ....grid import Grid
from ...jwst.rotation_and_shift.shift_correction import build_shift_correction
from ..data.observation import Observation


class PhaseShiftCorrection(jft.Model):
    """Phase shift correction to uv coordinates."""

    def __init__(
        self,
        shift: jft.Model,
        uvw: np.ndarray,
        freq: np.ndarray,
    ):
        """
        Initialize the CoordinatesCorrection model.

        Parameters
        ----------
        shift : jft.Model
            The distribution of the shift correction.
        uvw: np.ndarray
            The uvw coordiantes of the observation
        freq: np.ndarray,
            The frequencies of the observation.
        backend: Union[Ducc0Settings, FinufftSettings],
            The backend settings different units if Finfft.
        grid: Grid
            Needed for the Finufft backend.
        """
        self.shift = shift

        # NOTE : FROM FINUFFT response see `InterferometryResponseFinuFFT` in
        # re/response.py
        uvw = np.transpose((uvw[..., None] * freq / speedoflight.value), (0, 2, 1))
        # (pol, pos[m], freq[Hz])
        self.uvw = np.array([uvw[None, :, :, ii] for ii in range(3)])
        assert len(self.uvw[0].shape) == 3, "Check the polarization axis."

        super().__init__(domain=shift.domain)

    def __call__(self, params: dict) -> jnp.array:
        center_x, center_y = self.shift(params)
        # n = jnp.sqrt(1 - center_x**2 - center_y**2)
        return jnp.exp(
            -2j
            * np.pi
            * (
                self.uvw[0] * center_x + self.uvw[1] * center_y
                # + self.uvw[2] * (n - 1))
            )
        )


def build_phase_shift_correction_from_config(
    phase_shift_correction_config: CoordinatesCorrectionPriorConfig | None,
    observation: Observation,
    field_name: str,
):
    if phase_shift_correction_config is None:
        return None

    shift_correction_model = build_shift_correction(
        field_name,
        phase_shift_correction_config,
        unit=units.rad,
    )
    jft.logger.warning("Shift correction is ignoring the w-term")

    return PhaseShiftCorrection(
        shift_correction_model,
        observation.uvw,
        observation.freq,
    )
