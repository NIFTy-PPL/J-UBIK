# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %%

from typing import Callable, Union, Tuple, Optional

import nifty.re as jft
from astropy.coordinates import SkyCoord
from numpy.typing import ArrayLike

from .coordinates_correction import (
    build_coordinates_correction_model_from_grid, CoordinatesCorrection)
from .linear_rotation_and_shift import build_linear_rotation_and_shift
from .nufft_rotation_and_shift import build_nufft_rotation_and_shift
from .sparse_rotation_and_shift import build_sparse_rotation_and_shift
from ..reconstruction_grid import Grid
from ..wcs import (subsample_grid_centers_in_index_grid_non_vstack,
                   subsample_grid_corners_in_index_grid_non_vstack)
from ..wcs.wcs_base import WcsBase


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
        correction_model: Union[Callable, CoordinatesCorrection],
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
        correction_model : Union[Callable, CoordinatesCorrection]
            A model or function used to compute the correction to be applied to
            the sky coordinates.
        """
        assert isinstance(sky_domain, dict), ('Need to provide an internal key'
                                              'to the target of the sky model')

        self.sky_key = next(iter(sky_domain.keys()))
        self.correction_model = correction_model
        self.call = call

        correction_domain = correction_model.domain if isinstance(
            correction_model, CoordinatesCorrection) else {}
        super().__init__(domain=sky_domain | correction_domain)

    def __call__(self, x):
        return self.call(x[self.sky_key], self.correction_model(x))


def build_rotation_and_shift_model(
    sky_domain: dict,
    reconstruction_grid: Grid,
    world_extrema: Tuple[SkyCoord],
    data_grid_dvol: float,  # TODO: should this be for each data pixel, i.e. an array?
    data_grid_wcs: WcsBase,
    model_type: str,
    subsample: int,
    kwargs: dict,
    coordinate_correction: Optional[dict] = None,
) -> Callable[[Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]], ArrayLike]:
    """Rotation and shift model builder

    Parameters
    ----------
    sky_domain: dict
        Containing the sky_key and the shape_dtype of the reconstruction sky.

    world_extrema: Tuple[SkyCoord]
        The corners of the grid to be rotated and shifted into.

    reconstruction_grid: Grid
        The Grid underlying the reconstruction domain.

    data_grid_dvol: float
        The volume of the data pixel.

    data_grid_wcs: WcsBase
        The world coordinate system of the data grid.

    model_type: str
        The type of the rotation and shift model: (linear, nufft, sparse)

    subsample: int
        The subsample factor for the data grid. How many times a data pixel is
        subsampled in each direction.

    kwargs: dict
        linear:  dict, options
            - order: (0, 1), default: 1
            - sky_as_brightness: default: False

        sparse: dict, options
            - extend_factor, default: 1 (extension of the sky grid)
            - to_bottom_left: default: True (reconstruction in bottom
            left of extended grid)

        nufft: dict, options
            - sky_as_brightness: default: False

    coordinate_correction: dict
        domain_key: str
        priors: dict
            - shift: Mean and sigma for the Gaussian distribution of
            shift model.
            - rotation: Mean and sigma of the Gaussian distribution
            for theta [rad]


    Returns
    -------
    RotationAndShiftModel(dict(sky, correction)) -> rotated_and_shifted_sky
    """

    assert reconstruction_grid.dvol.unit == data_grid_dvol.unit

    correction_model = build_coordinates_correction_model_from_grid(
        coordinate_correction['domain_key'] if coordinate_correction is not None else None,
        coordinate_correction['priors'] if coordinate_correction is not None else None,
        data_grid_wcs,
        reconstruction_grid,
        subsample_grid_centers_in_index_grid_non_vstack(
            world_extrema,
            data_grid_wcs,
            reconstruction_grid.wcs,
            subsample)
    )

    match model_type:
        case 'linear':
            call = build_linear_rotation_and_shift(
                sky_dvol=reconstruction_grid.dvol.value,
                sub_dvol=data_grid_dvol.value / subsample**2,
                **kwargs.get('linear', dict(order=1, sky_as_brightness=False)),
            )

        case 'nufft':
            # TODO: check output shape
            out_shape = correction_model.target.shape[1:] if isinstance(
                correction_model, CoordinatesCorrection) else correction_model(None).shape[1:]

            call = build_nufft_rotation_and_shift(
                sky_dvol=reconstruction_grid.dvol.value,
                sub_dvol=data_grid_dvol.value / subsample**2,
                sky_shape=next(iter(sky_domain.values())).shape,
                out_shape=out_shape,
                **kwargs.get('nufft', dict(sky_as_brightness=False))
            )

        case 'sparse':
            # Sparse cannot update the coordinates, this is why the
            # correction_model is not passed to the builder.
            sparse_kwargs = kwargs.get('sparse', dict(
                extend_factor=1, to_bottom_left=False))
            call = build_sparse_rotation_and_shift(
                index_grid=reconstruction_grid.index_grid(**sparse_kwargs),
                subsample_corners=subsample_grid_corners_in_index_grid_non_vstack(
                    world_extrema,
                    data_grid_wcs,
                    reconstruction_grid.wcs,
                    subsample),
            )

            def correction_model(_): return None

        case _:
            raise NotImplementedError(
                f"{model_type} is not implemented. Available rotation_and_shift"
                f"methods are: (linear, nufft, sparse)"
            )

    return RotationAndShiftModel(
        sky_domain=sky_domain,
        call=call,
        correction_model=correction_model
    )
