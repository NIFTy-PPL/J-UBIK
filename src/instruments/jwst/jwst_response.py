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

from ...wcs.wcs_jwst_data import WcsJwstData
from ...wcs.wcs_astropy import WcsAstropy
from .integration.unit_conversion import build_unit_conversion
from .integration.integration import integration_factory
from .jwst_psf import build_psf_operator
from .masking.build_mask import build_mask
from .rotation_and_shift import RotationAndShift, build_rotation_and_shift
from .rotation_and_shift.coordinates_correction import (
    build_coordinates_corrected_from_grid,
    ShiftAndRotationCorrection,
)
from .zero_flux_model import build_zero_flux_model


class JwstResponse(jft.Model):
    """
    A that connects observational data to the corresponding sky and
    instrument models.

    This class models a data pipeline that includes rotation, shifting,
    PSF application, integration, and masking, with an optional zero-flux model.
    """

    def __init__(
        self,
        sky_domain: dict,
        rotation_and_shift: RotationAndShift,
        psf: Callable[[ArrayLike], ArrayLike],
        unit_conversion: Callable[[ArrayLike], ArrayLike],
        integrate: Callable[[ArrayLike], ArrayLike],
        zero_flux_model: jft.Model | None,
        mask: Callable[[ArrayLike], ArrayLike],
    ):
        """
        Initialize the DataModel with components for various data
        transformations.

        Parameters
        ----------
        sky_domain : dict
            A dictionary defining the sky domain, with a single key
            corresponding to the internal target of the sky model.
            This defines the input space of the data.
        rotation_and_shift : RotationAndShift | None
            A model that applies rotation and shift transformations
            to the input data.
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

        Raises
        ------
        AssertionError
            If `sky_domain` is not a dictionary or if it contains
            more than one key.
        """
        need_sky_key = "Need to provide an internal key to the target of the sky model"
        assert isinstance(sky_domain, dict), need_sky_key
        assert len(sky_domain.keys()) == 1, need_sky_key

        self.rotation_and_shift = rotation_and_shift
        self.psf = psf
        self.unit_conversion = unit_conversion
        self.integrate = integrate
        self.zero_flux_model = zero_flux_model
        self.mask = mask

        domain = sky_domain | rotation_and_shift.domain
        if zero_flux_model is not None:
            domain = domain | zero_flux_model.domain
        super().__init__(domain=domain)

    def __call__(self, x):
        out = self.rotation_and_shift(x)
        out = self.psf(out)
        out = self.unit_conversion(out)
        out = self.integrate(out)
        if self.zero_flux_model is not None:
            out = out + self.zero_flux_model(x)
        out = self.mask(out)
        return out


def build_jwst_response(
    sky_domain: dict,
    data_subsampled_centers: SkyCoord,
    data_meta: DataMetaInformation,
    sky_wcs: WcsAstropy,
    sky_meta: SkyMetaInformation,
    rotation_and_shift_algorithm: Union[LinearConfig, NufftConfig],
    shift_and_rotation_correction: ShiftAndRotationCorrection | None,
    psf: np.ndarray | None,
    zero_flux_prior_config: ProbabilityConfig | None,
    data_mask: ArrayLike | None,
) -> JwstResponse:
    """
    Builds the data model for a Jwst observation.

    The data model pipline:
    rotation_and_shift | psf | integrate | zero flux | mask

    Parameters
    ----------
    sky_domain: dict
        Containing the sky_key and the shape_dtype of the reconstruction sky.
    data_subsampled_centers: SkyCoord
        The world coordinates of the subsampled data. I.e. the world coordinates of the
        subsample centers of the data.
    data_meta: DataMetaInformation
        Meta information on the data, needed here:
            - unit
            - dvol
            - subsample
            - identifier
    sky_wcs: WcsAstropy
        The wcs of the sky/reconstruction grid.
    sky_meta: SkyMetaInformation
        Meta information on the sky, needed here:
            - unit
    rotation_and_shift_algorithm: Union[LinearConfig, NufftConfig],
        The information for the rotation_and_shift algorithm.
    shift_and_rotation_correction: CoordiantesCorrectionPriorConfig | None
        The prior for the shift and rotation coordinates correction.
    psf: np.ndarray
        The kernel of the psf as a np.ndarray.
    data_mask: ArrayLike
        The mask on the data
    sky_unit: astropy.units.Unit
        The unit fo the sky
    """

    need_sky_key = "Need to provide an internal key to the target of the sky model."
    assert isinstance(sky_domain, dict), need_sky_key

    coordinates = build_coordinates_corrected_from_grid(
        shift_and_rotation_correction=shift_and_rotation_correction,
        reconstruction_grid_wcs=sky_wcs,
        world_coordinates=data_subsampled_centers,
        indexing="ij",
        shift_only=True,
    )

    # non = build_coordinates_corrected_from_grid(
    #     shift_and_rotation_correction=None,
    #     reconstruction_grid_wcs=sky_wcs,
    #     world_coordinates=data_subsampled_centers,
    #     indexing="ij",
    # )
    # k = {k: np.zeros(v.shape) for k, v in coordinates.domain.items()}
    # exit()

    rotation_and_shift = build_rotation_and_shift(
        sky_domain=sky_domain,
        coordinates=coordinates,
        algorithm_config=rotation_and_shift_algorithm,
        indexing="ij",
    )

    unit_conversion = build_unit_conversion(
        sky_unit=sky_meta.unit,
        sky_dvol=sky_wcs.dvol,
        data_unit=data_meta.unit,
        data_dvol=data_meta.dvol / data_meta.subsample**2,
    )

    integrate = integration_factory(
        unit=data_meta.unit,
        high_resolution_shape=rotation_and_shift.target.shape,
        reduction_factor=data_meta.subsample,
    )

    psf = build_psf_operator(psf)

    zero_flux_model = build_zero_flux_model(
        data_meta.identifier, zero_flux_prior_config
    )

    mask = build_mask(data_mask)

    return JwstResponse(
        sky_domain=sky_domain,
        rotation_and_shift=rotation_and_shift,
        psf=psf,
        unit_conversion=unit_conversion,
        integrate=integrate,
        zero_flux_model=zero_flux_model,
        mask=mask,
    )
