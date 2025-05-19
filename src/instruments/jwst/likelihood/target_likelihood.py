from ....likelihood import build_gaussian_likelihood
from ....grid import Grid

from ..filter_projector import FilterProjector
from ..data.loader.target_loader import TargetData
from ..data.jwst_data import DataMetaInformation
from ..parse.jwst_response import SkyMetaInformation
from ..parse.rotation_and_shift.rotation_and_shift import LinearConfig, NufftConfig
from ..rotation_and_shift.coordinates_correction import (
    ShiftAndRotationCorrection,
)
from ..zero_flux_model import build_zero_flux_model
from ..parse.zero_flux_model import ZeroFluxPriorConfigs

from ..jwst_response import build_jwst_response, build_sky_to_subsampled_data

# Libraries
import jax.numpy as jnp
import nifty8.re as jft
import numpy as np


def build_target_likelihood_and_response(
    filter: str,
    grid: Grid,
    filter_projector: FilterProjector,
    target_data: TargetData,
    filter_meta: DataMetaInformation,
    sky_meta: SkyMetaInformation,
    rotation_and_shift_algorithm: LinearConfig | NufftConfig,
    shift_and_rotation_correction: ShiftAndRotationCorrection | None,
    zero_flux_prior_configs: ZeroFluxPriorConfigs,
):
    energy_name = filter_projector.get_key(filter_meta.color)
    sky_in_subsampled_data = build_sky_to_subsampled_data(
        sky_domain={energy_name: filter_projector.target[energy_name]},
        data_subsampled_centers=target_data.subsample_centers,
        sky_wcs=grid.spatial,
        rotation_and_shift_algorithm=rotation_and_shift_algorithm,
        shift_and_rotation_correction=shift_and_rotation_correction,
    )

    zero_flux_model = build_zero_flux_model(
        f"{filter}_target",
        zero_flux_prior_configs.get_name_setting_or_default(filter),
        shape=(target_data.data.shape[0], 1, 1),
    )

    jwst_target_response = build_jwst_response(
        sky_in_subsampled_data=sky_in_subsampled_data,
        data_meta=filter_meta,
        data_subsample=target_data.subsample,
        sky_meta=sky_meta,
        psf=np.array(target_data.psf),
        zero_flux_model=zero_flux_model,
        data_mask=np.array(target_data.mask),
    )

    likelihood_target = build_gaussian_likelihood(
        jnp.array(np.array(target_data.data)[np.array(target_data.mask)], dtype=float),
        jnp.array(np.array(target_data.std)[np.array(target_data.mask)], dtype=float),
        model=jwst_target_response,
    )

    return likelihood_target, jwst_target_response
