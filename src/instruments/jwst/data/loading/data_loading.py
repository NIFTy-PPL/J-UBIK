from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u
from nifty8.re import logger

# Parse
from ...parse.jwst_likelihoods import StarData, TargetData
from ...parse.jwst_psf import JwstPsfKernelConfig
from ...parse.masking.data_mask import ExtraMasks

from .....grid import Grid
from ...alignment.star_alignment import StarAlignment
from ...psf.jwst_kernel import load_psf_kernel
from ..jwst_data import JwstData
from ..preloading.data_bounds import DataBounds


def target_loading(
    target_data: TargetData,
    observation_id: int,
    jwst_data: JwstData,
    target_bounds: DataBounds,
    target_grid: Grid,
    extra_masks: ExtraMasks,
    psf_kernel_configs: JwstPsfKernelConfig,
):
    data, mask, std = jwst_data.bounding_data_mask_std_by_bounding_indices(
        target_bounds.bounds[observation_id],
        target_grid.spatial,
        extra_masks,
    )
    target_data.append_observation(
        meta=jwst_data.meta,
        data=data,
        mask=mask,
        std=std,
        subsample_centers=jwst_data.data_subpixel_centers(
            target_bounds.bounds[observation_id],
            subsample=target_data.subsample,
        ),
        psf=load_psf_kernel(
            jwst_data=jwst_data,
            subsample=target_data.subsample,
            target_center=target_grid.spatial.center,
            config_parameters=psf_kernel_configs,
        ),
    )


def star_alignment_loading(
    stars_data: dict[StarData],
    star_alignment: StarAlignment,
    observation_id: int,
    jwst_data: JwstData,
    extra_masks: ExtraMasks,
    psf_kernel_configs: JwstPsfKernelConfig,
):
    for star in star_alignment.get_stars(observation_id):
        fov_pixel = (
            star_alignment.alignment_meta.fov.to(u.arcsec)
            / jwst_data.meta.pixel_distance.to(u.arcsec)
        ).value
        fov_pixel = np.array((int(np.round(fov_pixel)),) * 2)
        if (fov_pixel % 2).sum() == 0:
            fov_pixel += 1

        bounding_indices = star.bounding_indices(jwst_data, fov_pixel)
        data, mask, std = jwst_data.bounding_data_mask_std_by_bounding_indices(
            row_minmax_column_minmax=bounding_indices,
            additional_masks_corners=extra_masks,
        )

        stars_data[star.id].append_observation(
            meta=jwst_data.meta,
            subsample=star_alignment.alignment_meta.subsample,
            data=data,
            mask=mask,
            std=std,
            psf=load_psf_kernel(
                jwst_data=jwst_data,
                subsample=star_alignment.alignment_meta.subsample,
                target_center=star.position,
                config_parameters=psf_kernel_configs,
            ),
            sky_array=np.zeros(
                [s * star_alignment.alignment_meta.subsample for s in data.shape]
            ),
            star_in_subsampled_pixles=star.pixel_position_in_subsampled_data(
                jwst_data.wcs,
                min_row=bounding_indices[0],
                min_column=bounding_indices[2],
                subsample_factor=star_alignment.alignment_meta.subsample,
            ),
            observation_id=observation_id,
        )


def data_loading(
    telescope_cfg: dict[str, Any],
    filepaths: list[Path],
    target_grid: Grid,
    star_alignment: StarAlignment | None,
    target_bounds: DataBounds,
    psf_kernel_configs: JwstPsfKernelConfig,
    extra_masks: ExtraMasks,
):
    # NOTE : Just for reference
    # telescope_cfg = cfg[telescope_key]
    target_data = TargetData.from_yaml_dict(telescope_cfg["target"])

    stars = star_alignment.get_stars() if star_alignment else None
    stars_data = {star.id: StarData() for star in stars} if star_alignment else None

    for observation_id, filepath in enumerate(filepaths):
        logger.info(f"Loading: {observation_id} {filepath}")

        jwst_data = JwstData(filepath)

        # import matplotlib.pyplot as plt
        # from functools import partial
        # from ...plotting.plotting_sky import plot_jwst_panels, plot_sky_coords
        #
        # fig, axes = plot_jwst_panels(
        #     [jwst_data.dm.data],
        #     [jwst_data.wcs],
        #     nrows=1,
        #     ncols=1,
        #     vmin=0.05,
        #     vmax=0.5,
        #     coords_plotter=partial(
        #         plot_sky_coords,
        #         sky_coords=[s.position for s in stars],
        #         marker_color="red",
        #         marker="x",
        #     ),
        # )
        # plt.show()

        target_loading(
            target_data,
            observation_id,
            jwst_data,
            target_bounds,
            target_grid,
            extra_masks,
            psf_kernel_configs,
        )

        if star_alignment:
            star_alignment_loading(
                stars_data=stars_data,
                star_alignment=star_alignment,
                observation_id=observation_id,
                jwst_data=jwst_data,
                extra_masks=extra_masks,
                psf_kernel_configs=psf_kernel_configs,
            )

    return target_data, stars, stars_data
