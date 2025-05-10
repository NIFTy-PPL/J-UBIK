# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %

from typing import Callable, Union

import nifty8.re as jft
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from numpy.typing import ArrayLike

from .parse.parametric_model.parametric_prior import ProbabilityConfig
from .parse.rotation_and_shift.rotation_and_shift import LinearConfig, NufftConfig
from .parse.jwst_response import SkyMetaInformation
from .data.jwst_data import DataMetaInformation

from .alignment.star_model import StarInData
from ...wcs.wcs_astropy import WcsAstropy
from .integration.unit_conversion import build_unit_conversion
from .integration.integration import integration_factory
from .psf.psf_learning import LearnablePsf
from .psf.psf_operator import build_psf_operator_strategy, PsfDynamic, PsfStatic
from .masking.build_mask import build_mask
from .rotation_and_shift import RotationAndShift, build_rotation_and_shift
from .rotation_and_shift.coordinates_correction import (
    build_coordinates_corrected_for_field,
    ShiftAndRotationCorrection,
)


class JwstResponse(jft.Model):
    """
    Linear response model that connects jwst observational data to the corresponding a
    sky model.

    Schematic pipeline:
    psf | unit_conversion | integrate | zero flux | mask
    """

    def __init__(
        self,
        sky_model: jft.Model | RotationAndShift | StarInData,
        psf: PsfStatic | PsfDynamic,
        unit_conversion: Callable[ArrayLike, ArrayLike],
        integrate: Callable[ArrayLike, ArrayLike],
        zero_flux_model: jft.Model | None,
        mask: Callable[ArrayLike, ArrayLike],
    ):
        """Initialize the Jwst response with different components of linear
        transformations.

        Parameters
        ----------
        sky_model : jft.Model | RotationAndShift | StarInData,
            A model has as output the sky in the frame of the data.
        psf : callable
            A function that applies a point spread function (PSF) to the
            input data.
        unit_conversion : callable
            A function that transforms the unit of the sky to the data unit.
        integrate : callable
            A function that performs integration on the input data.
        zero_flux_model : jft.Model | None
            A secondary model to account for zero flux.
            If provided, its output is added to the domain model's output.
        mask : callable
            A function that applies a mask to the final output.
        """

        self.sky_model = sky_model
        self.psf = psf
        self.unit_conversion = unit_conversion
        self.integrate = integrate
        self.zero_flux_model = zero_flux_model
        self.mask = mask

        domain = sky_model.domain | psf.domain[1]
        if zero_flux_model is not None:
            domain = domain | zero_flux_model.domain
        super().__init__(domain=domain)

    def __call__(self, x):
        out = self.sky_model(x)
        out = self.psf((out, x))
        out = self.unit_conversion(out)
        out = self.integrate(out)
        if self.zero_flux_model is not None:
            out = out + self.zero_flux_model(x)
        out = self.mask(out)
        return out


def build_sky_to_subsampled_data(
    sky_domain: dict[str, jft.ShapeWithDtype],
    data_subsampled_centers: SkyCoord | list[SkyCoord],
    sky_wcs: WcsAstropy,
    rotation_and_shift_algorithm: LinearConfig | NufftConfig,
    shift_and_rotation_correction: ShiftAndRotationCorrection | None,
) -> RotationAndShift:
    """Build the sky to subsampled data. The sky will be rotated and shifted by
    interpolation to the subsampled data center, whose central world coordinate is given
    by `data_subsampled_centers`. Additionally we can correct for shift and rotation
    errors.

    Parameters
    ----------
    sky_domain: dict
        The sky domain.
    data_subsampled_centers: SkyCoord
        The world coordinates of the subsampled data pixel centers.
    sky_wcs: WcsAstropy
        The wcs system of the sky (model target).
    rotation_and_shift_algorithm: LinearConfig | NufftConfig
        The interpolation algorithm, handling shift and rotation of data.
    shift_and_rotation_correction: ShiftAndRotationCorrection
        The probability distribution for the shift and rotation correction.
        If None, we assume that the data_subsampled_centers are correct.

    Return
    ------
    An operator that takes the output of sky model (saved in sky_domain) and projects
    the sky onto the - potentially subsampled - data pixel.
    """
    coordinates = build_coordinates_corrected_for_field(
        shift_and_rotation_correction=shift_and_rotation_correction,
        reconstruction_grid_wcs=sky_wcs,
        world_coordinates=data_subsampled_centers,
        indexing="ij",
        shift_only=True,
    )

    return build_rotation_and_shift(
        sky_domain=sky_domain,
        coordinates=coordinates,
        algorithm_config=rotation_and_shift_algorithm,
        indexing="ij",
    )


def build_jwst_response(
    sky_in_subsampled_data: jft.Model | RotationAndShift | StarInData,
    data_meta: DataMetaInformation,
    data_subsample: int,
    sky_meta: SkyMetaInformation,
    psf: np.ndarray | LearnablePsf | None,
    zero_flux_model: jft.Model | None,
    data_mask: ArrayLike | None,
) -> JwstResponse:
    """
    Builds the linear response of the Jwst to the sky. The sky must be in in the same
    coordinate frame as the - potentially subsampled - data.

    Schematic pipline:
    psf | unit_conversion | integrate | zero flux | mask

    Parameters
    ----------
    sky_in_subsampled_data_domain: ShapeWithDtype
        The shape and dtype of the sky in the subsampled data frame.
    data_meta: DataMetaInformation, needed here:
        - unit                  # Unit of the data
        - dvol                  # pixel volume of the data ~pixel_distance**2
        - pixel_distance        # 2-d distance between pixels
    data_subsample: int
        Subsample factor of the data
    sky_meta: SkyMetaInformation, needed here:
        - unit                  # Unit of the data
        - dvol                  # Pixel volume of the sky
    psf: np.ndarray
        The kernel of the psf as a np.ndarray.
    zero_flux_model : jft.Model
        The model for the a constant (zero) flux in the data.
    data_mask: ArrayLike
        The mask on the data
    """

    psf = build_psf_operator_strategy(sky_in_subsampled_data.target, psf)

    unit_conversion = build_unit_conversion(
        sky_unit=sky_meta.unit,
        sky_dvol=sky_meta.dvol,
        data_unit=data_meta.unit,
        data_dvol=data_meta.dvol / data_subsample**2,
    )

    integrate = integration_factory(
        unit=data_meta.unit,
        high_resolution_shape=sky_in_subsampled_data.target.shape,
        reduction_factor=data_subsample,
    )

    mask = build_mask(data_mask)

    return JwstResponse(
        sky_model=sky_in_subsampled_data,
        psf=psf,
        unit_conversion=unit_conversion,
        integrate=integrate,
        zero_flux_model=zero_flux_model,
        mask=mask,
    )
