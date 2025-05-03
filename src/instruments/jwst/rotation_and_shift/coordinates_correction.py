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
        domain_key: str,
        correction_prior: CoordinatesCorrectionPriorConfig,
        rotation_center: SkyCoord,
    ):
        self.shift = build_parametric_prior_from_prior_config(
            domain_key + "_shift",
            correction_prior.shift_in(correction_prior.shift_unit),
            correction_prior.shift.mean.shape,
            as_model=True,
        )
        self.shift_unit = correction_prior.shift_unit

        self.rotation_angle = build_parametric_prior_from_prior_config(
            domain_key + "_rotation",
            correction_prior.rotation_in(u.rad),
            correction_prior.rotation.mean.shape,
            as_model=True,
        )
        self.rotation_unit = u.rad

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
        self._1_over_pixel_distance = (
            1 / pixel_distance.to(shift_and_rotation.shift_unit).value
        )

        super().__init__(domain=self.shift_and_rotation.shift.domain)

    def __call__(self, params: dict) -> ArrayLike:
        shift = self.shift_and_rotation.shift(params) * self._1_over_pixel_distance
        return self._coordinates + shift[..., None, None]


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
            (jnp.cos(theta) * self._coords[0] - jnp.sin(theta) * self._coords[1])
            + self.rotation_center[0]
            + shift[0]
        )

        y = (
            (jnp.sin(theta) * self._coords[0] + jnp.cos(theta) * self._coords[1])
            + self.rotation_center[1]
            + shift[1]
        )
        return jnp.array((x, y)) * self._1_over_pixel_distance


def build_coordinates_corrected_from_grid(
    shift_and_rotation_correction: ShiftAndRotationCorrection | None,
    reconstruction_grid_wcs: WcsAstropy,
    world_coordinates: SkyCoord | list[SkyCoord],
    indexing: str = "ij",
    shift_only: bool = True,
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

    pixel_coordinates = np.array(
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
        return Coordinates(pixel_coordinates)

    shift_unit = shift_and_rotation_correction.shift_unit
    pixel_distance = reconstruction_grid_wcs.distances.to(shift_unit)

    if shift_only:
        return CoordinatesCorrectedShiftOnly(
            shift_and_rotation_correction, pixel_coordinates, pixel_distance
        )

    raise NotImplementedError

    # xxref, yyref = world_coordinates_to_index_grid(
    #     world_coordinates=world_coordinates,
    #     index_grid_wcs=reconstruction_grid_wcs,
    #     indexing=indexing,
    # )
    #
    # xx, yy = world_coordinates.spherical_offsets_to(
    #     shift_and_rotation_correction.rotation_center
    #     # reconstruction_grid_wcs.pixel_to_world(0, 0)
    # )
    #
    # xx_rot, yy_rot = reconstruction_grid_wcs.pixel_to_world(
    #     0.0, 0.0
    # ).spherical_offsets_to(shift_and_rotation_correction.rotation_center)
    #
    # xx, yy, xx_rot, yy_rot = (
    #     xx.to(shift_unit),
    #     yy.to(shift_unit),
    #     xx_rot.to(shift_unit),
    #     yy_rot.to(shift_unit),
    # )
    #
    # sep = haversine_sep(
    #     (world_coordinates.ra.rad, world_coordinates.dec.rad),
    #     (
    #         reconstruction_grid_wcs.pixel_to_world(0, 0).ra.rad,
    #         reconstruction_grid_wcs.pixel_to_world(0, 0).dec.rad,
    #     ),
    # )
    # sep = ((sep * u.rad).to(u.arcsec) / pixel_distance[0]).value
    # import matplotlib.pyplot as plt
    #
    # plt.imshow(sep - np.sqrt(xxref**2 + yyref**2))
    # plt.show()
    #
    # exit()
    #
    # return CoordinatesCorrected(
    #     shift_and_rotation_correction, (xx_rot, yy_rot), (xx, yy), pixel_distance
    # )


# def separation_small_angle(pos1: ArrayLike, pos2: ArrayLike):
#     ddec = pos2[1] - pos1[1]
#     dra = (pos2[0] - pos1[0]) * np.cos((pos1[1] + pos2[1]) / 2)
#     return dra, ddec
#
#
# def haversine_sep(coord1: ArrayLike, coord2: ArrayLike, *, radius: float = 1.0):
#     """
#     Great-circle separation using the haversine formula.
#
#     Parameters
#     ----------
#     coord1, coord2 : tuple or array-like, length 2
#         (ra, dec) of the two points.  ra is the first element (index 0),
#         dec the second (index 1).
#     radius : float, optional
#         Radius of the sphere whose surface distance is wanted.
#         Use 1.0 to get the separation in *radians*; use something
#         like `radius=206264.806` to get arc-seconds, or the
#         actual Earth radius to get metres/kilometres.
#     degrees : bool, optional
#         If True (default) the inputs are in degrees; otherwise in radians.
#
#     Returns
#     -------
#     distance : float
#         Surface distance = radius × central angle.
#         For `radius = 1` this is the central angle in radians.
#     """
#
#     lon1, lat1 = coord1  #  index 0  → ra / longitude
#     lon2, lat2 = coord2  #  index 1  → dec / latitude
#
#     dlon = lon2 - lon1
#     dlat = lat2 - lat1
#
#     # put Δλ into (−π, π] so that the shorter path is chosen
#     dlon = (dlon + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
#
#     sin_dlat2 = jnp.sin(dlat / 2.0)
#     sin_dlon2 = jnp.sin(dlon / 2.0)
#
#     hav = sin_dlat2**2 + jnp.cos(lat1) * jnp.cos(lat2) * sin_dlon2**2  # haversine
#     central_angle = 2.0 * jnp.arcsin(jnp.sqrt(hav))  # ρ  in rad
#
#     return radius * central_angle
