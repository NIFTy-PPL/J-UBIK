# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani


# Copyright(C) 2024 Max-Planck-Society

# %%

from typing import Callable, Union

import jax.numpy as jnp
import numpy as np
import nifty8.re as jft
from astropy import units as u
from astropy.coordinates import SkyCoord
from numpy.typing import ArrayLike

from .shift_correction import build_shift_correction
from ..parametric_model import build_parametric_prior_from_prior_config
from ....grid import Grid
from ....wcs.wcs_astropy import WcsAstropy
from ....wcs.wcs_jwst_data import WcsJwstData

from ..parse.rotation_and_shift.coordinates_correction import (
    CoordiantesCorrectionPriorConfig,
)


class ShiftAndRotationCorrection(jft.Model):
    def __init__(
        self,
        domain_key: str,
        correction_prior: CoordiantesCorrectionPriorConfig,
        rotation_center: SkyCoord,
    ):
        self.shift = build_shift_correction(
            domain_key, correction_prior, correction_prior.shift_unit
        )
        self.shift_unit = correction_prior.shift_unit

        self.rotation_angle = build_parametric_prior_from_prior_config(
            domain_key + "_rotation",
            correction_prior.rotation_in(u.rad),
            shape=(1,),
            as_model=True,
        )
        self.rotation_center = rotation_center

        super().__init__(domain=self.rotation_angle.domain | self.shift.domain)

    def __call__(self, x):
        return self.shift(x), self.rotation_angle(x)


class Coordinates:
    def __init__(self, coordiantes: ArrayLike):
        self.coordinates = coordiantes
        self.domain = {}

    def __call__(self, _):
        return self.coordinates


class CoordinatesCorrected(jft.Model):
    """
    Applies rotation and shift corrections to a set of coordinates.

    This model applies a rotation and shift to the given coordinates based on
    the provided priors for rotation and shift. The rotation is applied around
    a specified center, and the shift is scaled by the pixel distance.

    The transformation is defined as:
        ri = si * Rot(theta) * (pi - r)
           = Rot(theta) * (si * pi - si * r),
    where `si * r` is the rotation center.
    """

    def __init__(
        self,
        shift_and_rotation: ShiftAndRotationCorrection,
        rotation_center: tuple[int, int],
        coordinates: np.ndarray,
        pixel_distance: np.ndarray,
    ):
        """
        Initialize the ShiftAndRotationCorrection model.

        Parameters
        ----------
        shift : jft.Model
            A model that provides the prior distribution for the
            shift parameters.
        rotation : jft.Model
            A model that provides the prior distribution for the rotation angle.
        rotation_center : tuple of int
            The (x, y) coordinates around which the rotation is applied.
        coordinates : ArrayLike
            The coordinates to which the corrections will be applied. Assumed to be in
            units of the pixel distance.
        pixel_distance : np.ndarray
            The distances between pixels used to scale the shift values.
        """
        assert len(pixel_distance.shape) == 1

        self.shift_and_rotation = shift_and_rotation
        self.rotation_center = rotation_center
        self._coords = coordinates * pixel_distance[:, None, None]
        self._1_over_pixel_distance = 1 / pixel_distance[:, None, None]

        super().__init__(domain=self.shift_and_rotation.domain)

    def __call__(self, params: dict) -> ArrayLike:
        shift = self.shift_and_rotation.shift(params)
        theta = self.shift_and_rotation.rotation_angle(params)
        x = (
            (
                jnp.cos(theta) * (self._coords[0] - self.rotation_center[0])
                - jnp.sin(theta) * (self._coords[1] - self.rotation_center[1])
            )
            + self.rotation_center[0]
            + shift[0]
        )

        y = (
            (
                jnp.sin(theta) * (self._coords[0] - self.rotation_center[0])
                + jnp.cos(theta) * (self._coords[1] - self.rotation_center[1])
            )
            + self.rotation_center[1]
            + shift[1]
        )
        return jnp.array((x, y)) * self._1_over_pixel_distance


def build_coordinates_corrected_from_grid(
    shift_and_rotation_correction: ShiftAndRotationCorrection | None,
    reconstruction_grid_wcs: WcsAstropy,
    coordinates: ArrayLike,
) -> Union[Coordinates, CoordinatesCorrected]:
    """Build `CorrectedCorrdinates` from the grid and coordinates.

    If  coordiantes_correction` are None, it returns a lambda function that simply
    returns the original coordinates.

    Typically the shift correction is modeled as a Gaussian distribution:
        (x, y) ~ Gaussian(mean, sigma)

    The rotation correction is applied as:
        r_i = s_i * Rot(theta) * (p_i - r)
           = Rot(theta) * (s_i * p_i - s_i * r),
    where `s_i * r` represents the rotation center in the coordinate units
    (si * pi).

    Parameters
    ----------
    priors : CoordiantesCorrectionPriorConfig | None
        A dictionary containing the priors for shift and rotation.
        If None, no coordinate correction is applied and the return is a lambda
        function which returns the original coordinates.
    reconstruction_grid_wcs : WcsAstropy
        The grid used to define the coordinate system for the correction model.
    coordinates : ArrayLike
        The coordinates to be corrected by the model. Which are assumed to be given in
        pixel units/positions of the reconstruction grid.

    Returns
    -------
    Union[Callable[[dict, ArrayLike], ArrayLike], ShiftAndRotationCorrection]
        If `priors` is None, returns a lambda function that returns the
        original coordinates.
        Otherwise, returns an instance of `ShiftAndRotationCorrection` with
        the specified priors and parameters.

    """

    if shift_and_rotation_correction is None:
        return Coordinates(coordinates)

    rotation_center = np.array(
        reconstruction_grid_wcs.world_to_pixel(
            shift_and_rotation_correction.rotation_center
        )
    )
    pixel_distance = np.array(
        reconstruction_grid_wcs.distances_in(shift_and_rotation_correction.shift_unit)
    ).reshape(shift_and_rotation_correction.shift.target.shape)

    return CoordinatesCorrected(
        shift_and_rotation_correction, rotation_center, coordinates, pixel_distance
    )
