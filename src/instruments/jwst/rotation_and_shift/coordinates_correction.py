# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani


# Copyright(C) 2024 Max-Planck-Society

# %%

from typing import Optional, Callable, Union

import jax.numpy as jnp
import numpy as np
import nifty8.re as jft
from astropy import units as u
from jax.numpy import array
from numpy.typing import ArrayLike

from .shift_correction import build_shift_correction
from ..parametric_model import build_parametric_prior_from_prior_config
from ....grid import Grid
from ....wcs.wcs_astropy import WcsAstropy
from ....wcs.wcs_jwst_data import WcsJwstData

from ..parse.rotation_and_shift.coordinates_correction import (
    CoordiantesCorrectionPriorConfig,
)


class CoordinatesWithCorrection(jft.Model):
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
        shift: jft.Model,
        rotation: jft.Model,
        rotation_center: tuple[int, int],
        coords: ArrayLike,
    ):
        """
        Initialize the CoordinatesCorrection model.

        Parameters
        ----------
        shift : jft.Model
            A model that provides the prior distribution for the
            shift parameters.
        rotation : jft.Model
            A model that provides the prior distribution for the rotation angle.
        pix_distance : tuple of float
            The distances between pixels used to scale the shift values.
        rotation_center : tuple of int
            The (x, y) coordinates around which the rotation is applied.
        coords : ArrayLike
            The coordinates to which the corrections will be applied.
        """
        self.rotation_center = rotation_center
        self.shift = shift
        self.rotation = rotation
        self._coords = coords

        super().__init__(domain=rotation.domain | shift.domain)

    def __call__(self, params: dict) -> ArrayLike:
        shft = self.shift(params)
        theta = self.rotation(params)
        x = (
            (
                jnp.cos(theta) * (self._coords[0] - self.rotation_center[0])
                - jnp.sin(theta) * (self._coords[1] - self.rotation_center[1])
            )
            + self.rotation_center[0]
            + shft[0]
        )

        y = (
            (
                jnp.sin(theta) * (self._coords[0] - self.rotation_center[0])
                + jnp.cos(theta) * (self._coords[1] - self.rotation_center[1])
            )
            + self.rotation_center[1]
            + shft[1]
        )
        return jnp.array((x, y))


class Coordinates:
    def __init__(self, coordiantes: ArrayLike):
        self.coordinates = coordiantes

    def __call__(self, _):
        return self.coordinates


def build_coordinates_correction_from_grid(
    domain_key: str,
    priors: Optional[CoordiantesCorrectionPriorConfig],
    data_wcs: Union[WcsJwstData, WcsAstropy],
    reconstruction_grid: Grid,
    coords: ArrayLike,
) -> Union[Coordinates, CoordinatesWithCorrection]:
    """Builds a `CoordinatesCorrection` model based on a grid and WCS data.
    If priors are None, it returns a lambda function that simply
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
    domain_key : str
        A key used to generate names for the shift and rotation priors.
    priors : Optional[CoordiantesCorrectionPriorConfig]
        A dictionary containing the priors for shift and rotation.
        If None, no coordinate correction is applied and the return is a lambda
        function which returns the original coordinates.
    data_wcs : Union[WcsJwstData, WcsAstropy]
        The WCS data used to determine the reference pixel coordinates
        for the correction model.
    reconstruction_grid : Grid
        The grid used to define the coordinate system for the correction model.
    coords : ArrayLike
        The coordinates to be corrected by the model.

    Returns
    -------
    Union[Callable[[dict, ArrayLike], ArrayLike], CoordinatesCorrection]
        If `priors` is None, returns a lambda function that returns the
        original coordinates.
        Otherwise, returns an instance of `CoordinatesCorrection` with
        the specified priors and parameters.

    Raises
    ------
    NotImplementedError
        If `data_wcs` is of a type not supported by this function
        (i.e., not `WcsJwstData` or `WcsAstropy`).
    """

    if priors is None:
        return Coordinates(coords)

    header = data_wcs.to_header()

    if isinstance(data_wcs, WcsJwstData):
        rpix = (header["CRPIX1"],), (header["CRPIX2"],)
        rpix = data_wcs.index_to_world_location(rpix)
    elif isinstance(data_wcs, WcsAstropy):
        # FIXME: The following lines should be the same with the previous
        rpix = (header["CRPIX1"], header["CRPIX2"])
        rpix = data_wcs.index_to_world_location(rpix)[0]
    else:
        raise NotImplementedError(
            f"The type of world coordinate system {type(data_wcs)} is not "
            "supported. Supported types [WcsAstropy, WcsJwstData]."
        )

    rotation_center = np.array(
        reconstruction_grid.spatial.world_location_to_index(rpix)
    )

    shift = build_shift_correction(domain_key, priors)
    pix_distance = array(
        reconstruction_grid.spatial.distances_in(priors.shift_unit)
    ).reshape(shift.target.shape)

    rotation = build_parametric_prior_from_prior_config(
        domain_key + "_rotation",
        priors.rotation_in(u.rad),
        shape=(1,),
        as_model=True,
    )

    return CoordinatesWithCorrection(shift, rotation, rotation_center, coords)
