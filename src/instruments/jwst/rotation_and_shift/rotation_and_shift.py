# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %%

from typing import Callable, Union, Tuple

import nifty8.re as jft
from astropy.coordinates import SkyCoord
from numpy.typing import ArrayLike

from .coordinates_correction import CoordinatesWithCorrection
from .linear_rotation_and_shift import build_linear_rotation_and_shift
from .nufft_rotation_and_shift import build_nufft_rotation_and_shift
from .sparse_rotation_and_shift import build_sparse_rotation_and_shift
from ....grid import Grid
from ....wcs import subsample_grid_corners_in_index_grid_non_vstack
from ....wcs.wcs_base import WcsBase

from ..parse.rotation_and_shift.rotation_and_shift import (
    LinearConfig,
    NufftConfig,
    SparseConfig,
)


class RotationAndShiftModel(jft.Model):
    """
    A model for applying rotation and/or shift corrections to sky coordinates.

    This model adjusts the sky coordinates by applying a specified correction
    model and then using a callable function to transform the corrected
    coordinates.
    """

    def __init__(
        self,
        sky_domain: dict,
        call: Callable,
        # FIXME: This should only take ArrayLike !
        coordinates: Union[ArrayLike, Callable, CoordinatesWithCorrection],
    ):
        """
        Initialize the RotationAndShiftModel.

        Parameters
        ----------
        sky_domain : dict
            A dictionary specifying the domain for the sky coordinates.
            This should include the internal key for accessing the target
            of the sky model.
        call : callable
            A function that applies a callable transformation to the
            input data.
        coordinates : Union[ArrayLike, Callable, CoordinatesCorrection]
            The coordinates, which are possibly corrected by a model.
        """
        assert isinstance(sky_domain, dict), (
            "Need to provide an internal keyto the target of the sky model"
        )

        self.sky_key = next(iter(sky_domain.keys()))
        self.coordinates = coordinates
        if not callable(self.coordinates):
            self.coordinates = lambda _: coordinates
        self.call = call

        correction_domain = (
            coordinates.domain
            if isinstance(coordinates, CoordinatesWithCorrection)
            else {}
        )
        super().__init__(domain=sky_domain | correction_domain)

    def __call__(self, x):
        return self.call(x[self.sky_key], self.coordinates(x))


def _infere_output_shape_from_coordinates(
    coordinates: Union[ArrayLike, callable, CoordinatesWithCorrection],
):
    if isinstance(coordinates, CoordinatesWithCorrection):
        return coordinates.target.shape[1:]
    elif callable(coordinates):
        return coordinates(None).shape[1:]
    else:
        return coordinates.shape[1:]


def build_rotation_and_shift_model(
    sky_domain: dict,
    reconstruction_grid: Grid,
    world_extrema: Tuple[SkyCoord],
    data_grid_wcs: WcsBase,
    subsample: int,
    algorithm_config: Union[LinearConfig, NufftConfig, SparseConfig],
    coordinates: Union[ArrayLike, Callable, CoordinatesWithCorrection],
    indexing: str,
) -> RotationAndShiftModel:
    """Builds a RotationAndShiftModel according to the `algorithm_config`.

    Parameters
    ----------
    sky_domain: dict
        Containing the sky_key and the shape_dtype of the reconstruction sky.
    world_extrema: Tuple[SkyCoord]
        The corners of the grid to be rotated and shifted into.
    reconstruction_grid: Grid
        The Grid underlying the reconstruction domain.
    data_grid_wcs: WcsBase
        The world coordinate system of the data grid.
    algorithm_config: Union[LinearConfig, NufftConfig, SparseConfig]
        The type of the rotation and shift model: (linear, nufft, sparse)
    subsample: int
        The subsample factor for the data grid. How many times a data pixel is
        subsampled in each direction.
    coordinates: Union[ArrayLike, Callable, CoordinatesWithCorrection]
        The coordinates of the subsampled data. Precise: The coordinate center
        of the data pixel.

    Returns
    -------
    RotationAndShiftModel(dict(sky, correction)) -> rotated_and_shifted_sky
    """

    if isinstance(algorithm_config, LinearConfig):
        call = build_linear_rotation_and_shift(
            indexing=indexing,
            **vars(algorithm_config),
        )

    elif isinstance(algorithm_config, NufftConfig):
        # TODO: check output shape
        out_shape = _infere_output_shape_from_coordinates(coordinates)
        call = build_nufft_rotation_and_shift(
            sky_shape=reconstruction_grid.spatial.shape,
            out_shape=out_shape,
            indexing=indexing,
            **vars(algorithm_config),
        )

    elif isinstance(algorithm_config, SparseConfig):
        # TODO: Sparse cannot update the coordinates
        call = build_sparse_rotation_and_shift(
            index_grid=reconstruction_grid.spatial.index_grid(**vars(algorithm_config)),
            subsample_corners=subsample_grid_corners_in_index_grid_non_vstack(
                world_extrema, data_grid_wcs, reconstruction_grid.spatial, subsample
            ),
        )

    return RotationAndShiftModel(
        sky_domain=sky_domain, call=call, coordinates=coordinates
    )
