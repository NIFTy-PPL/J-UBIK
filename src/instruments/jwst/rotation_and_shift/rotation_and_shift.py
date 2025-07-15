# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %%

from typing import Callable, Union

import nifty.re as jft
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

    elif isinstance(domain, jft.ShapeWithDtype):
        assert typ == "coordinates"
        assert len(domain.shape) == 3, f"Unexpected shape for {typ}: {domain.shape}."
        return domain.shape[1:]

    raise ValueError


def build_rotation_and_shift(
    sky_domain: dict[str, jft.ShapeWithDtype],
    coordinates: Union[
        Coordinates, CoordinatesCorrectedShiftOnly, CoordinatesCorrectedShiftAndRotation
    ],
    algorithm_config: Union[LinearConfig, NufftConfig],
    indexing: str,
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

    Returns
    -------
    RotationAndShift(dict(sky, correction)) -> rotated_and_shifted_sky
    """

    if isinstance(algorithm_config, LinearConfig):
        rotation_and_shift_algorithm = build_linear_rotation_and_shift(
            indexing=indexing,
            **vars(algorithm_config),
        )

    elif isinstance(algorithm_config, NufftConfig):
        # TODO: check output shape
        rotation_and_shift_algorithm = build_nufft_rotation_and_shift(
            sky_shape=_infere_shape_from_domain(sky_domain, "sky"),
            out_shape=_infere_shape_from_domain(coordinates.target, "coordinates"),
            indexing=indexing,
            **vars(algorithm_config),
        )

    if len(coordinates.target.shape) == 3:
        pass  # Everything is fine
    elif len(coordinates.target.shape) == 4:
        rotation_and_shift_algorithm = vmap(rotation_and_shift_algorithm, (None, 0))
    else:
        raise ValueError(f"Coordinates have unknown shape: {coordinates.target.shape}'")

    return RotationAndShift(
        sky_domain=sky_domain,
        coordinates=coordinates,
        rotation_and_shift_algorithm=rotation_and_shift_algorithm,
    )
