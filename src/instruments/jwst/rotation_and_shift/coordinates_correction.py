# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani


# Copyright(C) 2024 Max-Planck-Society

# %%

from typing import Callable, Union

import jax.numpy as jnp
import numpy as np
import nifty.re as jft
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
    CorrectionModel,
)


class ShiftAndRotationCorrection(jft.Model):
    def __init__(
        self,
        domain_key: str,
        correction_prior: CoordinatesCorrectionPriorConfig,
        rotation_center: SkyCoord,
    ):
        self.model = correction_prior.model  # shift, rshift
        self.shift: jft.Model = build_parametric_prior_from_prior_config(
            domain_key + "_shift",
            correction_prior.shift_in(correction_prior.shift_unit),
            correction_prior.shift.mean.shape,
            as_model=True,
        )
        self.shift_unit: u.Unit = correction_prior.shift_unit

        self.rotation_angle: jft.Model = build_parametric_prior_from_prior_config(
            domain_key + "_rotation",
            correction_prior.rotation_in(u.rad),
            correction_prior.rotation.mean.shape,
            as_model=True,
        )
        self.rotation_unit: u.Unit = u.rad

        self.rotation_center: SkyCoord = rotation_center

        super().__init__(domain=self.rotation_angle.domain | self.shift.domain)

    def __call__(self, x):
        return self.shift(x), self.rotation_angle(x)


class Coordinates:
    def __init__(self, coordiantes: ArrayLike):
        self.coordinates = coordiantes
        self.domain = {}

    def __call__(self, _):
        return self.coordinates


def _get_reshape(coordinates: np.ndarray) -> tuple[Ellipsis]:
    shape_nochange = (Ellipsis,)  # equivalent to shift[...]
    shape_addaxes = (Ellipsis, None, None)  # equivalent to shift[..., None, None]

    if len(coordinates.shape) == 4:
        return shape_addaxes

    return shape_nochange


class CoordinatesCorrectedShiftOnly(jft.Model):
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
        pixel_coordinates: np.ndarray,
        pixel_distance: u.Quantity,
        shape: tuple[Ellipsis],
    ):
        """
        Initialize the ShiftAndRotationCorrection model.

        Parameters
        ----------
        shift_and_rotation : ShiftAndRotationCorrection
            A model that provides the prior distribution for the shift parameters.
        pixel_coordinates : np.ndarray
            The coordinates of the subsampled data pixels inside the pixel units of the
            reconstruction array (into which we need to index).
        pixel_distance : u.Quantity
            The distances between pixels in units of the reconstructed shift, used to
            transform the scale into pixel units.
        """
        assert len(pixel_distance) == 2

        self.shift_and_rotation = shift_and_rotation
        self._coordinates = pixel_coordinates
        self._shape = shape
        self._1_over_pixel_distance = (
            1 / pixel_distance.to(shift_and_rotation.shift_unit).value
        )

        super().__init__(domain=self.shift_and_rotation.shift.domain)

    def __call__(self, params: dict) -> ArrayLike:
        shift = self.shift_and_rotation.shift(params) * self._1_over_pixel_distance
        return self._coordinates + shift[self._shape]


class CoordinatesCorrectedShiftAndRotation(jft.Model):
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
        rotation_center: np.ndarray,
        pixel_coordinates: np.ndarray,
        pixel_distance: tuple[float, float],
        shape: tuple[Ellipsis],
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
        self._shape = shape
        self._coords = pixel_coordinates
        self._1_over_pixel_distance = (
            1 / pixel_distance.to(shift_and_rotation.shift_unit).value
        )

        super().__init__(domain=self.shift_and_rotation.domain)

    def __call__(self, params: dict) -> ArrayLike:
        shift = self.shift_and_rotation.shift(params)
        theta = self.shift_and_rotation.rotation_angle(params)[..., None]
        x = (
            (jnp.cos(theta) * self._coords[0] - jnp.sin(theta) * self._coords[1])
            + self.rotation_center[0][self._shape]
            + shift[0][self._shape]
        )

        y = (
            (jnp.sin(theta) * self._coords[0] + jnp.cos(theta) * self._coords[1])
            + self.rotation_center[1][self._shape]
            + shift[1][self._shape]
        )
        return jnp.array((x, y)) * self._1_over_pixel_distance[self._shape]


def build_coordinates_corrected_for_stars(
    shift_and_rotation_correction: ShiftAndRotationCorrection | None,
    pixel_coordinates: list[np.ndarray],
    pixel_distance: u.Quantity,
    observation_ids: tuple[int],
) -> Union[
    Coordinates, CoordinatesCorrectedShiftOnly, CoordinatesCorrectedShiftAndRotation
]:
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
        return Coordinates(pixel_coordinates)

    shift_unit = shift_and_rotation_correction.shift_unit
    pixel_distance = u.Quantity(
        (pixel_distance.to(shift_unit), pixel_distance.to(shift_unit))
    )

    if shift_and_rotation_correction.model == CorrectionModel.SHIFT:
        assert len(observation_ids) == pixel_coordinates.shape[0]

        if pixel_coordinates.shape == shift_and_rotation_correction.shift.target.shape:
            shape = (Ellipsis,)
        else:
            shape = (observation_ids, Ellipsis)

        return CoordinatesCorrectedShiftOnly(
            shift_and_rotation_correction, pixel_coordinates, pixel_distance, shape
        )

    raise NotImplementedError


def build_coordinates_corrected_for_field(
    shift_and_rotation_correction: ShiftAndRotationCorrection | None,
    reconstruction_grid_wcs: WcsAstropy,
    world_coordinates: SkyCoord | list[SkyCoord],
    indexing: str = "ij",
) -> Union[
    Coordinates, CoordinatesCorrectedShiftOnly, CoordinatesCorrectedShiftAndRotation
]:
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
    if not isinstance(world_coordinates, list):
        world_coordinates = [world_coordinates]

    fixed_coordinates = np.array(
        [
            world_coordinates_to_index_grid(
                world_coordinates=wc,
                index_grid_wcs=reconstruction_grid_wcs,
                indexing=indexing,
            )
            for wc in world_coordinates
        ]
    )

    if shift_and_rotation_correction is None:
        return Coordinates(fixed_coordinates)

    shift_unit = shift_and_rotation_correction.shift_unit
    reconstruction_scale = reconstruction_grid_wcs.distances.to(shift_unit)

    if shift_and_rotation_correction.model == CorrectionModel.SHIFT:
        assert (  # Check that we have the same amount of observations (length of first/0th axis)
            shift_and_rotation_correction.shift.target.shape
            == fixed_coordinates.shape[:2]
        )
        assert len(fixed_coordinates.shape) == 4

        shape = (Ellipsis, None, None)
        return CoordinatesCorrectedShiftOnly(
            shift_and_rotation_correction,
            fixed_coordinates,
            reconstruction_scale,
            shape,
        )

    elif shift_and_rotation_correction.model == CorrectionModel.RSHIFT:
        assert (  # Check that we have the same amount of observations (length of first/0th axis)
            shift_and_rotation_correction.shift.target.shape
            == fixed_coordinates.shape[:2]
        )
        assert len(fixed_coordinates.shape) == 4

        shape = (Ellipsis, None, None)

        jft.logger.info("WARNING: THIS SHOULDN't BE USED IN PRODUCTION")

        raise NotImplementedError

        return CoordinatesCorrectedShiftAndRotation(
            shift_and_rotation_correction,
            rotation_center=np.array([0, 0]),
            pixel_coordinates=fixed_coordinates,
            pixel_distance=reconstruction_scale,
            shape=shape,
        )

    # elif shift_and_rotation_correction.model == CorrectionModel.GENERAL:
    #     raise NotImplementedError

    else:
        raise NotImplementedError(
            "Unknown coordinate correction model: "
            f"{shift_and_rotation_correction.model}"
        )
