from .wcs.wcs_corners import get_pixel_corners
from .wcs.wcs_subsampling import get_subsamples_from_wcs
from .integration_model import (
    build_sparse_integration, build_sparse_integration_model,
    build_linear_integration, build_integration_model,
    build_nufft_integration)


def build_rotation_and_shift_model(
        reconstruction_grid,
        data_key,
        data_grid,
        data_mask,
        sky_model,
        data_model_keyword,
        subsample,
        updating=False):
    '''Build the rotation and shift model. The sky will be rotated and shifted
    from the reconstruction_grid onto the data_grid.

    '''

    sky_dvol = reconstruction_grid.dvol.value
    sub_dvol = data_grid.dvol.value / subsample**2,

    pixel_corners = get_pixel_corners(
        data_grid.world_extrema,
        data_grid.wcs,
        reconstruction_grid.wcs)

    subsample_centers = get_subsamples_from_wcs(
        data_grid.world_extrema,
        data_grid.wcs,
        reconstruction_grid.wcs,
        subsample)

    if data_model_keyword == 'sparse':
        sparse_matrix = build_sparse_integration(
            reconstruction_grid.index_grid(),
            pixel_corners,
            data_mask)
        return build_sparse_integration_model(
            sparse_matrix, sky_model)

    elif data_model_keyword == 'linear':
        linear = build_linear_integration(
            sky_dvol,
            sub_dvol,
            subsample_centers,
            data_mask,
            order=1,
            updating=updating)
        return build_integration_model(linear, sky_model)

    elif data_model_keyword == 'nufft':
        nufft = build_nufft_integration(
            sky_dvol=sky_dvol,
            sub_dvol=sub_dvol,
            subsample_centers=subsample_centers,
            mask=data_mask,
            sky_shape=reconstruction_grid.shape,
        )
        return build_integration_model(nufft, sky_model)


def build_data_model(
        reconstruction_grid,
        data_key,
        data_grid,
        data_mask,
        sky_model,
        data_model_keyword,
        subsample,
        updating=False):

    return build_rotation_and_shift_model(
        reconstruction_grid,
        data_key,
        data_grid,
        data_mask,
        sky_model,
        data_model_keyword,
        subsample,
        updating)
