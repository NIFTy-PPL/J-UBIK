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
from ....wcs import world_coordinates_to_index_grid

from ..parse.rotation_and_shift.coordinates_correction import (
    CoordinatesCorrectionPriorConfig,
)


class ShiftAndRotationCorrection(jft.Model):
    def __init__(
        self,
        domain_key: str | list[str],
        correction_prior: CoordinatesCorrectionPriorConfig
        | list[CoordinatesCorrectionPriorConfig],
        rotation_center: SkyCoord | list[SkyCoord],
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
        self.rotation_unit = u.rad
        self.rotation_center = rotation_center

        super().__init__(domain=self.rotation_angle.domain | self.shift.domain)

    def __call__(self, x):
        return self.shift(x), self.rotation_angle(x)


def build_shift_and_rotation_correction(
    domain_key: str | list[str],
    correction_prior: CoordinatesCorrectionPriorConfig
    | list[CoordinatesCorrectionPriorConfig],
    rotation_center: SkyCoord | list[SkyCoord],
):
    if not isinstance(domain_key, list):
        domain_key = [domain_key]

    if isinstance(correction_prior, list):
        shift_unit = correction_prior[0].shift_unit
        for cp in correction_prior:
            assert shift_unit == cp.shift_unit
    else:
        correction_prior = [correction_prior]

    assert len(domain_key) == len(rotation_center) == len(correction_prior)

    # shift
    shift_unit = correction_prior[0].shift_unit
    rotation_center = SkyCoord(rotation_center)
    shifts = []
    rotations = []
    exit()
    for dk, cp in zip(domain_key, correction_prior):
        shifts.append(build_shift_correction(dk + "_shift", cp, shift_unit))
        rotations.append(
            build_rotation_correction(
                dk + "_rotation", cp.rotation_in(u.rad), shape=(1,), as_model=True
            )
        )

    # rotation_angle
    #
    # return ShiftAndRotationCorrection(
    #     shift, shift_unit, rotation_angle, rotation_center
    # )


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
        rotation_center: tuple[u.Quantity, u.Quantity],
        coordinates: tuple[u.Quantity, u.Quantity],
        pixel_distance: tuple[float, float],
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
        self.rotation_center = [rc.value for rc in rotation_center]
        self._coords = [coords.value for coords in coordinates]
        self._1_over_pixel_distance = 1 / pixel_distance[:, None, None]

        super().__init__(domain=self.shift_and_rotation.domain)

    def __call__(self, params: dict) -> ArrayLike:
        shift = self.shift_and_rotation.shift(params)
        theta = self.shift_and_rotation.rotation_angle(params)
        x = (
            jnp.cos(theta) * (self._coords[0] + shift[0])
            - jnp.sin(theta) * (self._coords[1] + shift[1])
        ) + self.rotation_center[0]

        y = (
            jnp.sin(theta) * (self._coords[0] + shift[0])
            + jnp.cos(theta) * (self._coords[1] + shift[1])
        ) + self.rotation_center[1]
        return jnp.array((x, y)) * self._1_over_pixel_distance


def build_coordinates_corrected_from_grid(
    shift_and_rotation_correction: ShiftAndRotationCorrection | None,
    reconstruction_grid_wcs: WcsAstropy,
    world_coordinates: SkyCoord,
    indexing: str = "ij",
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
    coordinates : SkyCoord
        The world coordinates of the (subsampled) data pixel, whose position can be
        corrected.

    Returns
    -------
    Union[Callable[[dict, ArrayLike], ArrayLike], ShiftAndRotationCorrection]
        If `priors` is None, returns a lambda function that returns the
        original coordinates.
        Otherwise, returns an instance of `ShiftAndRotationCorrection` with
        the specified priors and parameters.

    """

    if shift_and_rotation_correction is None:
        pixel_coordinates = world_coordinates_to_index_grid(
            world_coordinates=world_coordinates,
            index_grid_wcs=reconstruction_grid_wcs,
            indexing=indexing,
        )
        return Coordinates(pixel_coordinates)

    shift_unit = shift_and_rotation_correction.shift_unit

    rr = world_coordinates.separation(shift_and_rotation_correction.rotation_center)
    phi = world_coordinates.position_angle(
        shift_and_rotation_correction.rotation_center
    )
    xx, yy = (rr * (np.cos(phi), np.sin(phi))).to(shift_unit)

    rr = shift_and_rotation_correction.rotation_center.separation(
        reconstruction_grid_wcs.center
    )
    phi = shift_and_rotation_correction.rotation_center.position_angle(
        reconstruction_grid_wcs.center
    )
    xx_rot, yy_rot = (rr * (np.cos(phi), np.sin(phi))).to(shift_unit)

    pixel_distance = np.array(reconstruction_grid_wcs.distances_in(shift_unit))

    return CoordinatesCorrected(
        shift_and_rotation_correction, (xx_rot, yy_rot), (xx, yy), pixel_distance
    )
