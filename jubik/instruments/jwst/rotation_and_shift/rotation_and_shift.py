# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %%

from typing import Callable, Union

import nifty.re as jft
import numpy as np
from astropy import units as u
from jax import vmap

from ..parse.rotation_and_shift.rotation_and_shift import (
    LinearConfig,
    NufftConfig,
)
from .coordinates_correction import (
    Coordinates,
    CoordinatesCorrectedShiftAndRotation,
    CoordinatesCorrectedShiftOnly,
)
from .linear_rotation_and_shift import build_linear_rotation_and_shift
from .nufft_rotation_and_shift import build_nufft_rotation_and_shift


class RotationAndShift(jft.Model):
    """
    A model for applying rotation and/or shift corrections to sky coordinates.

    This model adjusts the sky coordinates by applying a specified correction
    model and then using a callable function to transform the corrected
    coordinates.
    """

    def __init__(
        self,
        sky_domain: dict[str, jft.ShapeWithDtype],
        coordinates: Union[
            Coordinates,
            CoordinatesCorrectedShiftOnly,
            CoordinatesCorrectedShiftAndRotation,
        ],
        rotation_and_shift_algorithm: Callable,
        # FIXME: This should only take ArrayLike !
    ):
        """
        Initialize the RotationAndShift.

        Parameters
        ----------
        sky_domain : dict
            A dictionary specifying the domain for the sky coordinates.
            This should include the internal key for accessing the target
            of the sky model.
        rotation_and_shift_algorithm : callable
            A function that applies a callable transformation to the
            input data.
        coordinates : Union[ArrayLike, Callable, CoordinatesCorrection]
            The coordinates, which are possibly corrected by a model.
        """
        assert isinstance(sky_domain, dict), (
            "Need to provide an internal keyto the target of the sky model"
        )

        self._sky_key = next(iter(sky_domain.keys()))
        self.coordinates = coordinates
        self.rotation_and_shift_algorithm = rotation_and_shift_algorithm

        super().__init__(domain=sky_domain | coordinates.domain)

    def __call__(self, x):
        return self.rotation_and_shift_algorithm(x[self._sky_key], self.coordinates(x))


def _infere_shape_from_domain(
    domain: Union[jft.ShapeWithDtype, dict[str, jft.ShapeWithDtype]], typ: str
):
    if isinstance(domain, dict):
        assert typ == "sky"
        assert len(domain) == 1
        sky = next(iter(domain.values()))
        assert len(sky.shape) == 2, f"Unexpected shape for {typ}: {sky.shape}."

        return sky.shape

    elif typ == "coordinates" and hasattr(domain, "shape"):
        shape = domain.shape
        assert len(shape) in (3, 4), (
            f"Unexpected shape for {typ}: {shape}."
        )
        assert shape[-3] == 2, (
            f"Unexpected coordinate axis for {typ}: {shape}."
        )
        return shape[-2:]

    raise ValueError


def _points_inside_with_slack(coordinates, sky_shape) -> int:
    """The points inside the sky at the correction's zero latent, plus 5%: the
    NUFFT's slots for the points it interpolates (it falls back to every point
    when a correction moves more inside)."""
    centers = coordinates(jft.zeros_like(coordinates.domain))
    centers = np.moveaxis(np.asarray(centers), -3, 0)
    upper = np.array(sky_shape).reshape(2, *(1,) * (centers.ndim - 1))
    inside = np.all((centers >= 0) & (centers <= upper), axis=0)
    # at least one slot: a NUFFT of zero points is untried on the GPU
    return max(1, int(np.ceil(1.05 * inside.sum())))


def build_rotation_and_shift(
    sky_domain: dict[str, jft.ShapeWithDtype],
    coordinates: Union[
        Coordinates, CoordinatesCorrectedShiftOnly, CoordinatesCorrectedShiftAndRotation
    ],
    algorithm_config: Union[LinearConfig, NufftConfig],
    sky_pixel_scales_yx: u.Quantity | None = None,
) -> RotationAndShift:
    """Builds a RotationAndShift according to the `algorithm_config`.

    Parameters
    ----------
    sky_domain: dict
        Containing the sky_key and the shape_dtype of the reconstruction sky.
    algorithm_config: Union[LinearConfig, NufftConfig]
        The type of the rotation and shift model: (linear, nufft)
    coordinates: Union[ArrayLike, Callable, CoordinatesCorrectedShiftOnly, CoordinatesCorrectedShiftAndRotation]
        The coordinates of the subsampled data. Precise: The coordinate center
        of the data pixel.
    sky_pixel_scales_yx: astropy.units.Quantity | None
        The sky's pixel scales, to read a NUFFT `band_limit` (cycles per arcsec)
        in cycles per sky pixel. Needed only with a band limit.

    Returns
    -------
    RotationAndShift(dict(sky, correction)) -> rotated_and_shifted_sky
    """

    if isinstance(algorithm_config, LinearConfig):
        rotation_and_shift_algorithm = build_linear_rotation_and_shift(
            out_shape=_infere_shape_from_domain(coordinates.target, "coordinates"),
            **vars(algorithm_config),
        )

    elif isinstance(algorithm_config, NufftConfig):
        band_limit = algorithm_config.band_limit
        if band_limit == "auto":
            raise ValueError(
                "resolve a NUFFT band_limit of 'auto' for the filter first"
            )
        if band_limit is not None:
            if sky_pixel_scales_yx is None:
                raise ValueError("a NUFFT band_limit needs the sky's pixel scales")
            band_limit = tuple(band_limit * sky_pixel_scales_yx.to_value(u.arcsec))
        sky_shape = _infere_shape_from_domain(sky_domain, "sky")
        # TODO: check output shape
        rotation_and_shift_algorithm = build_nufft_rotation_and_shift(
            sky_shape=sky_shape,
            out_shape=_infere_shape_from_domain(coordinates.target, "coordinates"),
            mode=algorithm_config.mode,
            band_limit=band_limit,
            capacity=_points_inside_with_slack(coordinates, sky_shape),
        )

    if len(coordinates.target.shape) == 3:
        pass  # Everything is fine
    elif len(coordinates.target.shape) == 4:
        if isinstance(algorithm_config, LinearConfig):  # nufft takes the stack itself
            rotation_and_shift_algorithm = vmap(rotation_and_shift_algorithm, (None, 0))
    else:
        raise ValueError(f"Coordinates have unknown shape: {coordinates.target.shape}'")

    return RotationAndShift(
        sky_domain=sky_domain,
        coordinates=coordinates,
        rotation_and_shift_algorithm=rotation_and_shift_algorithm,
    )
