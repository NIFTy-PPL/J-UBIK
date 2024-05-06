from .wcs.wcs_corners import get_pixel_corners
from .wcs.wcs_subsampling import get_subsamples_from_wcs
import nifty8.re as jft
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from .mock_plotting import build_plot, build_evaluation_mask

import jax
jax.config.update('jax_platform_name', 'cpu')


def plot_square(ax, xy_positions, color='red'):
    # Convert to a NumPy array for easier manipulation
    xy_positions = np.array(xy_positions)

    maxx, maxy = np.argmax(xy_positions, axis=1)
    minx, miny = np.argmin(xy_positions, axis=1)

    square_corners = np.array([
        xy_positions[:, maxx],
        xy_positions[:, maxy],
        xy_positions[:, minx],
        xy_positions[:, miny],
        xy_positions[:, maxx],
    ])

    ax.plot(square_corners[:, 0], square_corners[:, 1],
            color=color,
            linestyle='-',
            linewidth=2)
    ax.scatter(*square_corners.T, color=color)


def downscale_sum(high_res_array, reduction_factor):
    """
    Sums the entries of a high-resolution array into a lower-resolution array
    by the given reduction factor.

    Parameters:
    - high_res_array: np.ndarray, the high-resolution array to be downscaled.
    - reduction_factor: int, the factor by which to reduce the resolution.

    Returns:
    - A lower-resolution array where each element is the sum of a block from the
      high-resolution array.
    """
    # Ensure the reduction factor is valid
    if high_res_array.shape[0] % reduction_factor != 0 or high_res_array.shape[1] % reduction_factor != 0:
        raise ValueError(
            "The reduction factor must evenly divide both dimensions of the high_res_array.")

    # Reshape and sum
    new_shape = (high_res_array.shape[0] // reduction_factor, reduction_factor,
                 high_res_array.shape[1] // reduction_factor, reduction_factor)
    return high_res_array.reshape(new_shape).sum(axis=(1, 3))


def create_mocksky(key, mshape, mdist, model_setup):
    offset, fluctuations = model_setup
    cfm = jft.CorrelatedFieldMaker(prefix='mock')
    cfm.set_amplitude_total_offset(**offset)
    cfm.add_fluctuations(
        mshape, mdist, **fluctuations, non_parametric_kind='power')
    mock_diffuse = cfm.finalize()
    return jnp.exp(mock_diffuse(jft.random_like(key, mock_diffuse.domain)))


def create_data(
    mock_grid,
    mock_sky,
    data_center,
    rota_shape,
    rota_fov,
    data_shape,
    rotation,
    full_info=False
):
    from jwst_handling.reconstruction_grid import Grid
    from jwst_handling.integration_models import build_nufft_integration

    rota_grid = Grid(
        data_center, shape=rota_shape, fov=rota_fov, rotation=rotation)
    data_grid = Grid(
        data_center, shape=data_shape, fov=rota_fov, rotation=rotation)

    interpolation_points = mock_grid.wcs.index_from_wl(
        rota_grid.wl_coords())[0]

    mask = np.full(rota_shape, True)
    nufft = build_nufft_integration(
        1, 1, interpolation_points[::-1, :, :][None], mask, mock_grid.shape)

    rota_sky = np.zeros(rota_shape)
    rota_sky[mask] = nufft(mock_sky)

    downscale = [r//d for r, d in zip(rota_shape, data_shape)]
    assert downscale[0] == downscale[1]
    data = downscale_sum(rota_sky, downscale[0])

    if full_info:
        return data_grid, data, rota_sky

    return data_grid, data


def setup(
    mock_key,
    rotation,
    repo_rotation,
    shift,
    repo_shift,
    reco_shape,
    mock_shape=1024,
    rota_shape=768,
    data_shape=48,
    plot=False,
):

    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from jwst_handling.reconstruction_grid import Grid

    DISTANCE = 0.05

    cx, cy = 0, 0
    CENTER = SkyCoord(cx*u.rad, cy*u.rad)

    # True sky
    MOCK_SHAPE = (mock_shape,)*2
    MOCK_DIST = (DISTANCE, DISTANCE)
    MOCK_FOV = [MOCK_SHAPE[ii]*(MOCK_DIST[ii]*u.arcsec) for ii in range(2)]
    mock_grid = Grid(CENTER, MOCK_SHAPE, MOCK_FOV)

    # Reco sky
    RECO_SHAPE = (reco_shape,)*2
    RECO_FOV = MOCK_FOV
    reco_grid = Grid(CENTER, RECO_SHAPE, RECO_FOV)
    comp_down = [r//d for r, d in zip(MOCK_SHAPE, RECO_SHAPE)]

    offset = dict(offset_mean=0.1, offset_std=[0.1, 0.05])
    fluctuations = dict(fluctuations=[0.3, 0.03],
                        loglogavgslope=[-3., 1.],
                        flexibility=[0.8, 0.1],
                        asperity=[0.2, 0.1])

    mock_sky = create_mocksky(
        mock_key, MOCK_SHAPE, MOCK_DIST, (offset, fluctuations))
    comparison_sky = downscale_sum(mock_sky, comp_down[0])

    # DATA SETUP
    ROTA_SHAPE = (rota_shape,)*2
    DATA_SHAPE = (data_shape,)*2
    rotation = rotation if isinstance(rotation, list) else [rotation]
    datas = {}
    for ii, (rot, repo_rot, shft, repo_shft) in enumerate(zip(rotation, repo_rotation,  shift, repo_shift)):
        # Intermediate rotated sky
        ROTATION = rot*u.deg
        ROTA_FOV = [ROTA_SHAPE[ii]*(MOCK_DIST[ii]*u.arcsec) for ii in range(2)]

        # Data sky
        DATA_DIST = [r//d*MOCK_DIST[0] for r, d in zip(ROTA_SHAPE, DATA_SHAPE)]

        data_center = SkyCoord(cx*u.rad+(shft[0]*u.arcsec).to(u.rad),
                               cy*u.rad+(shft[1]*u.arcsec).to(u.rad))
        data_grid, data, rota_sky = create_data(
            mock_grid, mock_sky, data_center, ROTA_SHAPE, ROTA_FOV, DATA_SHAPE,
            ROTATION, full_info=True)

        # UPDATE REPORTED CENTER, AND ROTATION.
        data_grid = Grid(
            SkyCoord((repo_shft[0]*u.arcsec).to(u.rad),
                     (repo_shft[1]*u.arcsec).to(u.rad)),
            shape=data_grid.shape,
            fov=data_grid.fov,
            rotation=repo_rot*u.deg)

        datas[f'd_{ii}'] = dict(data=data, grid=data_grid)

        if plot:
            arr = mock_grid.wcs.index_from_wl(data_grid.world_extrema).T

            fig, ax = plt.subplots(2, 2)
            (a00, a01), (a10, a11) = ax
            ims = []
            ims.append(a00.imshow(mock_sky, origin='lower'))
            plot_square(a00, arr)
            ims.append(a01.imshow(rota_sky, origin='lower'))
            ims.append(a10.imshow(comparison_sky, origin='lower'))
            ims.append(a11.imshow(data, origin='lower'))

            a00.set_title(
                f'Underlying sky; sh={MOCK_SHAPE[0]}, dist={MOCK_DIST[0]}')
            a01.set_title(f'Cut out; sh={ROTA_SHAPE[0]}, dist={MOCK_DIST[0]}')
            a10.set_title(
                f'Comparison sky; sh={RECO_SHAPE[0]}, dist={comp_down[0]*MOCK_DIST[0]}')
            a11.set_title(f'data sky; sh={DATA_SHAPE[0]}, dist={DATA_DIST[0]}')

            for a, i in zip(ax.flatten(), ims):
                plt.colorbar(i, ax=a, shrink=0.7)
            plt.show()

    return comparison_sky, reco_grid, datas


def build_sky_model(sky_key, shape, dist):

    offset = dict(offset_mean=3.7, offset_std=[0.1, 0.05])
    fluctuations = dict(fluctuations=[0.7, 0.03],
                        loglogavgslope=[-4.8, 1.],
                        flexibility=[0.8, 0.1],
                        asperity=[0.2, 0.1])

    cfm = jft.CorrelatedFieldMaker(prefix='reco')
    cfm.set_amplitude_total_offset(**offset)
    cfm.add_fluctuations(
        [int(shp) for shp in shape], dist,
        **fluctuations, non_parametric_kind='power')
    log_diffuse = cfm.finalize()

    def diffuse(x):
        return jnp.exp(log_diffuse(x))
    diffuse = jft.wrap_left(diffuse, sky_key)

    return jft.Model(diffuse, domain=log_diffuse.domain)


def sky_model_check(
    key,
    sky_model,
    comp_sky,
):

    m = sky_model(jft.random_like(key, sky_model.domain))

    fig, axis = plt.subplots(1, 3)
    im0 = axis[0].imshow(comp_sky, origin='lower')
    im1 = axis[1].imshow(m, origin='lower')
    im2 = axis[2].imshow(comp_sky-m, origin='lower', cmap='RdBu_r')
    axis[0].set_title('sky')
    axis[1].set_title('model')
    axis[2].set_title('residual')
    for im, ax in zip([im0, im1, im2], axis):
        fig.colorbar(im, ax=ax, shrink=0.7)
    plt.show()


def build_shift_model(key, mean_sigma):
    from charm_lensing.models.parametric_models.parametric_prior import (
        build_prior_operator)
    distribution_model_key = ('normal', *mean_sigma)
    shape = (2,)

    shift_model = build_prior_operator(key, distribution_model_key, shape)
    domain = {key: jft.ShapeWithDtype((shape))}
    return jft.Model(shift_model, domain=domain)


def build_data_model(
        reco_grid,
        data_key,
        data_grid,
        data_mask,
        sky_key,
        sky_model,
        data_model_keyword,
        subsample,
        updating=False):
    # FIXME: Better distribute the sky_model field to the operators (per key)
    from jwst_handling.integration_models import (
        build_sparse_integration, build_sparse_integration_model,
        build_linear_integration, build_integration_model,
        build_nufft_integration)

    sky_dvol = reco_grid.dvol.value
    sub_dvol = data_grid.dvol.value / subsample**2,

    pixel_corners = get_pixel_corners(
        data_grid.world_extrema,
        data_grid.wcs,
        reco_grid.wcs)

    subsample_centers = get_subsamples_from_wcs(
        data_grid.world_extrema,
        data_grid.wcs,
        reco_grid.wcs,
        subsample)

    if data_model_keyword == 'sparse':
        sparse_matrix = build_sparse_integration(
            reco_grid.index_grid(),
            pixel_corners,
            data_mask)
        return build_sparse_integration_model(sparse_matrix, sky_model, sky_key)

    elif data_model_keyword == 'linear':
        linear = build_linear_integration(
            sky_dvol,
            sub_dvol,
            subsample_centers,
            data_mask,
            order=1,
            updating=updating)
        if updating:
            raise NotImplementedError
        else:
            return build_integration_model(linear, sky_model, sky_key)

    elif data_model_keyword == 'nufft':
        nufft = build_nufft_integration(
            sky_dvol=sky_dvol,
            sub_dvol=sub_dvol,
            subsample_centers=subsample_centers,
            mask=data_mask,
            sky_shape=reco_grid.shape,
        )
        return build_integration_model(nufft, sky_model, sky_key)
