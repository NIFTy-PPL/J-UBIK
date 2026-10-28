# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani


# Copyright(C) 2024 Max-Planck-Society

# %%

from typing import Union

import jax.numpy as jnp
import nifty.re as jft
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from numpy.typing import ArrayLike

from ....models.distributions import build_parametric_prior_from_prior_config
from ....wcs.wcs_astropy import WcsAstropy
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


def _get_reshape(coordinates: np.ndarray) -> tuple[Ellipsis]:
    shape_nochange = (Ellipsis,)  # equivalent to shift[...]
    shape_addaxes = (Ellipsis, None, None)  # equivalent to shift[..., None, None]

    if len(coordinates.shape) == 4:
        return shape_addaxes

    return shape_nochange


class Coordinates(jft.Model):
    def __init__(self, coordiantes: ArrayLike):
        self.coordinates = coordiantes
        super().__init__(
            domain={},
            target=jft.ShapeWithDtype(coordiantes.shape, coordiantes.dtype),
        )

    def __call__(self, _):
        return self.coordinates


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


class CoordinatesCorrectedStarShift(jft.Model):
    """Star positions in subsampled data pixels, corrected by the shared shift.

    The shared shift moves the data sampling positions on the reconstruction grid
    (see `build_coordinates_corrected_for_field`). A source therefore appears
    displaced the opposite way: `p = p0 - J @ shift`, with `J` the change of the
    star's data pixel position per arcsec of shift along the grid (row, column).

    The optional `position` is the star's true position minus its Gaia position,
    in arcsec along the grid (row, column). It moves the star itself, so it enters
    with the opposite sign: `p = p0 - J @ (shift - position)`. A star that sits at
    Gaia plus `position` on the grid appears at `+ J @ position` in the data.
    """

    def __init__(
        self,
        shift_and_rotation: ShiftAndRotationCorrection,
        pixel_coordinates: np.ndarray,
        shift_jacobian: np.ndarray,
        observation_ids: np.ndarray,
        position: jft.Model | None = None,
    ):
        self.shift_and_rotation = shift_and_rotation
        self.position = position
        self._coordinates = pixel_coordinates
        self._arcsec_per_unit = u.Quantity(1, shift_and_rotation.shift_unit).to_value(
            u.arcsec
        )
        self._jacobian = shift_jacobian
        self._observation_ids = np.asarray(observation_ids)
        domain = self.shift_and_rotation.shift.domain
        if position is not None:
            domain = domain | position.domain
        super().__init__(domain=domain)

    def __call__(self, params: dict) -> ArrayLike:
        shift = self.shift_and_rotation.shift(params)[self._observation_ids]
        shift = shift * self._arcsec_per_unit
        if self.position is not None:
            shift = shift - self.position(params)
        return self._coordinates - jnp.einsum("nij,nj->ni", self._jacobian, shift)


def build_coordinates_corrected_for_stars(
    shift_and_rotation_correction: ShiftAndRotationCorrection | None,
    pixel_coordinates: np.ndarray,
    shift_jacobian: np.ndarray,
    observation_ids: tuple[int],
    position: jft.Model | None = None,
) -> Union[Coordinates, CoordinatesCorrectedStarShift]:
    """Build the star positions in the subsampled data pixels, `xy` indexed.

    Parameters
    ----------
    shift_and_rotation_correction : ShiftAndRotationCorrection | None
        The shared shift model. If None, the positions are returned uncorrected.
    pixel_coordinates : np.ndarray
        The star position in each exposure, shape `(n_exposures, 2)`.
    shift_jacobian : np.ndarray
        Change of the star position (subsampled pixels) per arcsec of shift of the
        sampling positions along the grid (row, column), shape `(n_exposures, 2, 2)`.
    observation_ids : tuple[int]
        The observation (shift row) of each exposure.
    position : jft.Model | None
        The star's true minus Gaia position in arcsec along the grid (row, column),
        shape `(2,)`, shared by all exposures. If None, the star sits at Gaia.
    """
    if shift_and_rotation_correction is None:
        if position is not None:
            raise ValueError("A star position offset needs a shift correction.")
        return Coordinates(pixel_coordinates)

    if shift_and_rotation_correction.model == CorrectionModel.SHIFT:
        assert len(observation_ids) == pixel_coordinates.shape[0]
        return CoordinatesCorrectedStarShift(
            shift_and_rotation_correction,
            pixel_coordinates,
            shift_jacobian,
            observation_ids,
            position,
        )

    raise NotImplementedError


def build_coordinates_corrected_for_field(
    shift_and_rotation_correction: ShiftAndRotationCorrection | None,
    reconstruction_grid_wcs: WcsAstropy,
    world_coordinates: SkyCoord | list[SkyCoord],
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
            reconstruction_grid_wcs.world_to_indices_yx(wc)
            for wc in world_coordinates
        ]
    )

    if shift_and_rotation_correction is None:
        return Coordinates(fixed_coordinates)

    shift_unit = shift_and_rotation_correction.shift_unit
    reconstruction_scale = reconstruction_grid_wcs.pixel_scales_yx.to(shift_unit)

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
