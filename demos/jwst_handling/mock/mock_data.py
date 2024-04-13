import nifty8.re as jft
import jax.numpy as jnp
import numpy as np

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


def create_data_old(key, shapes, mock_dist, model_setup, show=True):
    mock_shape, reco_shape,  data_shape = shapes
    mock_sky = create_mocksky(key, mock_shape, mock_dist, model_setup)

    comparison_sky = downscale_sum(mock_sky, mock_shape[0] // reco_shape[0])
    data = downscale_sum(mock_sky, mock_shape[0] // data_shape[0])
    mask = np.full(data_shape, True, dtype=bool)

    if show:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3)
        ims = []
        ims.append(axes[0].imshow(mock_sky, origin='lower'))
        ims.append(axes[1].imshow(comparison_sky,
                   origin='lower'))
        ims.append(axes[2].imshow(data, origin='lower'))
        axes[0].set_title('Mock sky')
        axes[1].set_title('Comparison sky')
        axes[2].set_title('Data')
        for im, ax in zip(ims, axes):
            fig.colorbar(im, ax=ax, shrink=0.7)
        plt.show()

    return mock_sky, comparison_sky, data, mask


def create_mocksky(key, mshape, mdist, model_setup):
    offset, fluctuations = model_setup
    cfm = jft.CorrelatedFieldMaker(prefix='mock')
    cfm.set_amplitude_total_offset(**offset)
    cfm.add_fluctuations(
        mshape, mdist, **fluctuations, non_parametric_kind='power')
    mock_diffuse = cfm.finalize()
    return jnp.exp(mock_diffuse(jft.random_like(key, mock_diffuse.domain)))


def create_data(
    center,
    mock_grid,
    mock_sky,
    rota_shape,
    rota_fov,
    data_shape,
    rotation,
    full_info=False
):
    from jwst_handling.reconstruction_grid import Grid
    from jwst_handling.integration_models import build_nufft_integration

    rota_grid = Grid(center, shape=rota_shape, fov=rota_fov, rotation=rotation)
    data_grid = Grid(center, shape=data_shape, fov=rota_fov, rotation=rotation)

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


def setup(mock_key, plot=False):
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from jwst_handling.reconstruction_grid import Grid

    DISTANCE = 0.05

    CENTER = SkyCoord(0*u.rad, 0*u.rad)

    # True sky
    MOCK_SHAPE = (1024, 1024)
    MOCK_DIST = (DISTANCE, DISTANCE)
    MOCK_FOV = [MOCK_SHAPE[ii]*(MOCK_DIST[ii]*u.arcsec) for ii in range(2)]
    mock_grid = Grid(CENTER, MOCK_SHAPE, MOCK_FOV)

    # Reco sky
    RECO_SHAPE = (128, 128)
    RECO_FOV = MOCK_FOV
    reco_grid = Grid(CENTER, RECO_SHAPE, RECO_FOV)
    comp_down = [r//d for r, d in zip(MOCK_SHAPE, RECO_SHAPE)]

    # Intermediate rotated sky
    ROTATION = 5*u.deg
    ROTA_SHAPE = (768, 768)
    ROTA_FOV = [ROTA_SHAPE[ii]*(MOCK_DIST[ii]*u.arcsec) for ii in range(2)]

    # Data sky
    DATA_SHAPE = (48, 48)

    offset = dict(offset_mean=0.1, offset_std=[0.1, 0.05])
    fluctuations = dict(fluctuations=[0.3, 0.03],
                        loglogavgslope=[-3., 1.],
                        flexibility=[0.8, 0.1],
                        asperity=[0.2, 0.1])

    mock_sky = create_mocksky(
        mock_key, MOCK_SHAPE, MOCK_DIST, (offset, fluctuations))
    comparison_sky = downscale_sum(mock_sky, comp_down[0])

    data_grid, data, rota_sky = create_data(
        CENTER, mock_grid, mock_sky, ROTA_SHAPE, ROTA_FOV, DATA_SHAPE,
        ROTATION, full_info=True)

    if plot:
        import matplotlib.pyplot as plt
        arr = mock_grid.wcs.index_from_wl(data_grid.world_extrema).T

        fig, ax = plt.subplots(1, 3)
        ax, ay, az = ax
        ax.imshow(mock_sky, origin='lower')
        plot_square(ax, arr)
        ay.imshow(rota_sky, origin='lower')
        az.imshow(data, origin='lower')
        plt.show()

    return comparison_sky, reco_grid, data, data_grid


if __name__ == '__main__':
    from jax import random
    import matplotlib.pyplot as plt
    from jwst_handling.integration_models import (build_sparse_integration,
                                                  build_linear_integration)

    key = random.PRNGKey(87)
    key, mock_key, noise_key, rec_key = random.split(key, 4)
    comp_sky, reco_grid, data, data_grid = setup(mock_key, plot=True)

    # Reconstruction Setup
    SUBSAMPLE = 3
    NOISE_SCALE = 0.01

    std = data.mean() * NOISE_SCALE
    d = data + random.normal(noise_key, data.shape, dtype=data.dtype) * std

    # For interpolation
    wl_data_subsample_centers = data_grid.wcs.wl_subsample_centers(
        data_grid.world_extrema, SUBSAMPLE)
    px_reco_subsample_centers = reco_grid.wcs.index_from_wl(
        wl_data_subsample_centers)

    # plt.imshow(comp_sky, origin='lower')
    # plt.scatter(*px_reco_subsample_centers[0])
    # plt.show()

    # For sparse
    wl_data_centers, (e00, e01, e10, e11) = data_grid.wcs.wl_pixelcenter_and_edges(
        data_grid.world_extrema)
    dpixcenter_in_rgrid = reco_grid.wcs.index_from_wl(wl_data_centers)[0]
    px_reco_index_edges = reco_grid.wcs.index_from_wl(
        [e00, e01, e11, e10])  # needs to be circular for sparse builder

    likelihood_info = dict(
        mask=np.full(data.shape, True),
        index_edges=px_reco_index_edges,
        index_subsample_centers=px_reco_subsample_centers,
        data=d,
        std=std
    )

    sparse_matrix = build_sparse_integration(
        reco_grid.index_grid(), likelihood_info['index_edges'], likelihood_info['mask'])
