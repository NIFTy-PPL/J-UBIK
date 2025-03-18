# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %%

from typing import Tuple

import nifty8.re as jft
from jax.numpy import array
from numpy.typing import ArrayLike

from ..parametric_model import build_parametric_prior


class ShiftModel(jft.Model):
    """
    A model that applies a shift correction to coordinates based on a
    prior distribution.

    The shift correction is performed by adding a shift vector to the input
    coordinates. The shift vector is sampled  from a prior distribution
    specified by `prior_model`.
    The shift is scaled by the `pix_distance`, which represents the size
    of an individual pixel.

    The transformation applied is:
        corrected_coords = coords + (shift / pix_distance)
    """

    def __init__(self, prior_model: jft.Model, pix_distance: Tuple[float]):
        """
        Initialize the ShiftModel.

        Parameters
        ----------
        prior_model : jft.Model
            A model that provides the prior distribution for the
            shift parameters.
        pix_distance : Tuple[float]
            The distance between pixels in the x and y directions.
        """
        self.pix_distance = pix_distance
        self.prior_model = prior_model

        # FIXME: HACK, the target shape is actually prescribed by the coords.
        # However, one does not necessarily know the coordinates shape upon
        # shift model creation.
        super().__init__(domain=prior_model.domain, target=prior_model.target)

    def __call__(self, params: dict, coords: ArrayLike) -> ArrayLike:
        return coords + self.prior_model(params) / self.pix_distance


def build_shift_model(
    domain_key: str, mean_sigma: Tuple[float], pix_distances: Tuple[float]
) -> ShiftModel:
    """
    The shift model is a Gaussian distribution over the positions (x, y).
    (x, y) <- Gaussian(mean, sigma)

    Parameters
    ----------
    domain_key: str
        The domain key for the shift model.
    mean_sigma: Tuple[float]
        The mean and sigma for the Gaussian distribution of shift model.
    pix_distances: Tuple[float]
        The distance between pixels in the x and y directions.

    Returns
    -------
    ShiftModel
        The shift model.
    """
    shape = (2, 1, 1)
    pix_distance = array(pix_distances).reshape(shape)

    # Build Prior
    shift_prior = build_parametric_prior(domain_key, ("normal", *mean_sigma), shape)
    shift_prior_model = jft.Model(
        shift_prior, domain={domain_key: jft.ShapeWithDtype(shape)}
    )

    return ShiftModel(shift_prior_model, pix_distance)
