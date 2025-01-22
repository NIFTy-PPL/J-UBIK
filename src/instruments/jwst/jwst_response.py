# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %

from .integration_model import build_sum
from .jwst_psf import build_psf_operator
from .rotation_and_shift import (
    build_rotation_and_shift_model, RotationAndShiftModel)
from .rotation_and_shift.coordinates_correction import (
    build_coordinates_correction_from_grid)
from .zero_flux_model import build_zero_flux_model
from .masking.build_mask import build_mask
from .wcs import subsample_grid_centers_in_index_grid_non_vstack

from .parse.parametric_model.parametric_prior import ProbabilityConfig
from .parse.rotation_and_shift.coordinates_correction import CoordiantesCorrectionPriorConfig

from typing import Callable, Optional

import nifty8.re as jft
from numpy.typing import ArrayLike


class JwstResponse(jft.Model):
    """
    A that connects observational data to the corresponding sky and
    instrument models.

    This class models a data pipeline that includes rotation, shifting,
    PSF application, integration, transmission correction, and masking,
    with an optional zero-flux model.
    """

    def __init__(
        self,
        sky_domain: dict,
        rotation_and_shift: Optional[RotationAndShiftModel],
        psf: Callable[[ArrayLike], ArrayLike],
        integrate: Callable[[ArrayLike], ArrayLike],
        transmission: float,
        zero_flux_model: Optional[jft.Model],
        mask: Callable[[ArrayLike], ArrayLike]
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
        rotation_and_shift : RotationAndShiftModel, optional
            A model that applies rotation and shift transformations
            to the input data.
        psf : callable
            A function that applies a point spread function (PSF) to the
            input data.
        integrate : callable
            A function that performs integration on the input data.
        transmission : float
            A transmission factor by which the output data is multiplied.
        zero_flux_model : jft.Model, optional
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
        need_sky_key = ('Need to provide an internal key to the target of the '
                        'sky model')
        assert isinstance(sky_domain, dict), need_sky_key
        assert len(sky_domain.keys()) == 1, need_sky_key

        self.rotation_and_shift = rotation_and_shift
        self.psf = psf
        self.integrate = integrate
        self.transmission = transmission
        self.zero_flux_model = zero_flux_model
        self.mask = mask

        domain = sky_domain | rotation_and_shift.domain
        if zero_flux_model is not None:
            domain = domain | zero_flux_model.domain
        super().__init__(domain=domain)

    def __call__(self, x):
        out = self.rotation_and_shift(x)
        out = self.psf(out)
        out = self.integrate(out)
        out = out * self.transmission
        if self.zero_flux_model is not None:
            out = out + self.zero_flux_model(x)
        out = self.mask(out)
        return out


def build_jwst_response(
    sky_domain: dict,
    data_identifier: str,
    data_subsample: int,
    rotation_and_shift_kwargs: Optional[dict],
    shift_and_rotation_correction_prior: Optional[CoordiantesCorrectionPriorConfig],
    psf_kernel: Optional[ArrayLike],
    transmission: float,
    zero_flux_prior_config: Optional[ProbabilityConfig],
    data_mask: Optional[ArrayLike],
) -> JwstResponse:
    """
    Builds the data model for a Jwst observation.

    The data model pipline:
    rotation_and_shift | psf | integrate | mask

    Parameters
    ----------
    sky_domain: dict
        Containing the sky_key and the shape_dtype of the reconstruction sky.

    data_subsample: int
        The subsample factor for the data grid.

    rotation_and_shift_kwargs: dict
        reconstruction_grid: Grid
        data_dvol: Unit, the volume of a data pixel
        data_wcs: WcsBase,
        data_model_type: str,
        world_extrema: Tuple[SkyCoord]

    shift_and_rotation_correction: Optional[CoordiantesCorrectionPriorConfig]
        The prior for the shift and rotation coordinates correction.

    psf_kernel_model:
        camera: str, NIRCam or MIRI
        filter: str
        center_pix: tuple, pixel according to which to evaluate the psf model
        webbpsf_path: str
        fov_pixels: Optional[int], how many pixles considered for the psf
        fov_arcsec: Optional[float], the arcseconds for the psf evaluation

    data_mask: ArrayLike
        The mask on the data
    """

    need_sky_key = ('Need to provide an internal key to the target of the sky '
                    'model.')
    assert isinstance(sky_domain, dict), need_sky_key

    world_extrema = rotation_and_shift_kwargs['world_extrema']
    reconstruction_grid = rotation_and_shift_kwargs['reconstruction_grid']
    data_wcs = rotation_and_shift_kwargs['data_wcs']

    coordinates = build_coordinates_correction_from_grid(
        f'{data_identifier}_correction',
        priors=shift_and_rotation_correction_prior,
        data_wcs=data_wcs,
        reconstruction_grid=reconstruction_grid,
        coords=subsample_grid_centers_in_index_grid_non_vstack(
            world_extrema=world_extrema,
            to_be_subsampled_grid_wcs=data_wcs,
            index_grid_wcs=reconstruction_grid.spatial,
            subsample=data_subsample)
    )

    rotation_and_shift = build_rotation_and_shift_model(
        sky_domain=sky_domain,
        reconstruction_grid=reconstruction_grid,
        world_extrema=world_extrema,
        data_grid_dvol=rotation_and_shift_kwargs['data_dvol'],
        data_grid_wcs=data_wcs,
        algorithm_config=rotation_and_shift_kwargs['algorithm_config'],
        subsample=data_subsample,
        coordinates=coordinates,
    )

    integrate = build_sum(
        high_res_shape=rotation_and_shift.target.shape,
        reduction_factor=data_subsample,
    )

    psf = build_psf_operator(psf_kernel)

    zero_flux_model = build_zero_flux_model(
        data_identifier, zero_flux_prior_config)

    mask = build_mask(data_mask)

    return JwstResponse(
        sky_domain=sky_domain,
        rotation_and_shift=rotation_and_shift,
        psf=psf,
        integrate=integrate,
        transmission=transmission,
        zero_flux_model=zero_flux_model,
        mask=mask
    )
