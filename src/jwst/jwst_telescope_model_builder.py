from .wcs.wcs_subsample_corners import subsample_grid_corners_in_index_grid
from .wcs.wcs_subsample_centers import subsample_grid_centers_in_index_grid
from .integration_model import (
    build_sparse_integration, build_sparse_integration_model,
    build_linear_integration, build_integration_model,
    build_nufft_integration)

from numpy import squeeze


def build_rotation_and_shift_model(
        reconstruction_grid,
        data_key,
        data_grid,
        data_mask,
        sky_model,
        data_model_keyword,
        subsample,
        updating=False):
    pass


def build_integration(
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

    pixel_corners = subsample_grid_corners_in_index_grid(
        data_grid.world_extrema,
        data_grid.wcs,
        reconstruction_grid.wcs,
        1)
    pixel_corners = squeeze(pixel_corners)

    subsample_centers = subsample_grid_centers_in_index_grid(
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

    return build_integration(
        reconstruction_grid,
        data_key,
        data_grid,
        data_mask,
        sky_model,
        data_model_keyword,
        subsample,
        updating)


# if __name__ == '__main__':
#     from astropy.coordinates import SkyCoord
#     import astropy.units as u

#     from .reconstruction_grid import Grid

#     reco_grid = Grid(SkyCoord(0*u.rad, 0*u.rad), (32,)*2, (0.1*u.arcsec,)*2)
#     data_grid = Grid(SkyCoord(0*u.rad, 0*u.rad), (16,)*2, (0.2*u.arcsec,)*2)

#     lin = build_integration(
#         reco_grid,
#         'data',
#         data_grid,
#         data_mask,)
