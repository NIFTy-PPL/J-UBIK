from ..wcs import subsample_grid_corners_in_index_grid
from ..wcs import subsample_grid_centers_in_index_grid

from .linear_integration import build_linear_integration
from .sparse_integration import build_sparse_integration
from .nufft_integration import build_nufft_integration

from numpy import squeeze


import nifty8.re as jft

from jax.experimental.sparse import BCOO

from typing import Callable, Union, Tuple, Optional
from numpy.typing import ArrayLike as AL


def build_integration_model(
    integration: Callable[Union[AL, Tuple[AL, AL]], AL],
    sky: jft.Model,
    shift: Optional[jft.Model] = None
):
    # FIXME: Update such that the integration model takes a key (SKY_INTERNAL)
    # or index for the multifrequency model?

    domain = sky.target.copy()
    sky_key = next(iter(sky.target.keys()))

    if shift is None:
        return jft.Model(
            lambda x: integration(x[sky_key]),
            domain=jft.Vector(domain)
        )

    domain.update(shift.domain)
    return jft.Model(
        lambda x: integration((x[sky_key], shift(x))),
        domain=jft.Vector(domain)
    )


def build_sparse_integration_model(
    sparse_matrix: BCOO,
    sky: jft.Model,
):
    domain = sky.target
    sky_key = next(iter(sky.target.keys()))

    return jft.Model(lambda x: sparse_matrix @ (x[sky_key]).reshape(-1),
                     domain=jft.Vector(domain))


def build_integration(
    reconstruction_grid,
    data_grid,
    data_mask,
    sky_model,
    data_model_keyword,
    subsample,
    updating=False
):
    '''Build the rotation and shift model. The sky will be rotated and shifted
    from the reconstruction_grid onto the data_grid.

    '''

    sky_dvol = reconstruction_grid.dvol.value
    sub_dvol = data_grid.dvol.value / subsample**2,

    pixel_corners = subsample_grid_corners_in_index_grid(
        data_grid.world_extrema(),
        data_grid.wcs,
        reconstruction_grid.wcs,
        1)
    pixel_corners = squeeze(pixel_corners)

    subsample_centers = subsample_grid_centers_in_index_grid(
        data_grid.world_extrema(),
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
