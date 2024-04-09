import nifty8.re as jft
import jax.numpy as jnp
import numpy as np

import jax
jax.config.update('jax_platform_name', 'cpu')


def plot_square(ax, xy_positions, color='r'):
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

    ax.plot(square_corners[:, 1], square_corners[:, 0],
            color=color,
            linestyle='-',
            linewidth=2)


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
        mock_shape, mock_dist, **fluctuations, non_parametric_kind='power')
    mock_diffuse = cfm.finalize()
    return jnp.exp(mock_diffuse(jft.random_like(key, mock_diffuse.domain)))


def create_data(
    center,
    mock_grid,
    mock_sky,
    rota_shape,
    rota_fov,
    data_shape,
    rotation
):

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

    return data_grid, data


def setup(mock_key):
    DISTANCE = 0.05

    CENTER = SkyCoord(0*u.rad, 0*u.rad)
    MOCK_SHAPE = (1024, 1024)
    MOCK_DIST = (DISTANCE, DISTANCE)
    MOCK_FOV = [MOCK_SHAPE[ii]*(MOCK_DIST[ii]*u.arcsec) for ii in range(2)]
    mock_grid = Grid(CENTER, MOCK_SHAPE, MOCK_FOV)

    RECO_SHAPE = (128, 128)
    RECO_FOV = MOCK_FOV
    reco_grid = Grid(CENTER, RECO_SHAPE, RECO_FOV)
    comp_down = [r//d for r, d in zip(MOCK_SHAPE, RECO_SHAPE)]

    ROTATION = 5*u.deg
    ROTA_SHAPE = (768, 768)
    ROTA_FOV = [ROTA_SHAPE[ii]*(MOCK_DIST[ii]*u.arcsec) for ii in range(2)]
    DATA_SHAPE = (48, 48)

    offset = dict(offset_mean=0.1, offset_std=[0.1, 0.05])
    fluctuations = dict(fluctuations=[0.3, 0.03],
                        loglogavgslope=[-3., 1.],
                        flexibility=[0.8, 0.1],
                        asperity=[0.2, 0.1])

    mock_sky = create_mocksky(
        mock_key, MOCK_SHAPE, MOCK_DIST, (offset, fluctuations))
    comparison_sky = downscale_sum(mock_sky, comp_down[0])

    data_grid, data = create_data(
        CENTER, mock_grid, mock_sky, ROTA_SHAPE, ROTA_FOV, DATA_SHAPE, ROTATION)

    return comparison_sky, reco_grid, data, data_grid


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from jax import random
    from charm_lensing.spaces import Space
    from charm_lensing.models.parametric_models.rotation import rotation as rotation_func
    from jwst_handling.integration_models import (
        build_nufft_integration
    )

    from jwst_handling.reconstruction_grid import Grid
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    key = random.PRNGKey(87)
    DISTANCE = 0.05
    DSHAPE = 64
    RSHAPE = 256
    key, mock_key, noise_key, rec_key = random.split(key, 4)
    mock_shape, mock_dist = (1024, 1024), (DISTANCE, DISTANCE)

    offset = dict(offset_mean=0.1, offset_std=[0.1, 0.05])
    fluctuations = dict(fluctuations=[0.3, 0.03],
                        loglogavgslope=[-3., 1.],
                        flexibility=[0.8, 0.1],
                        asperity=[0.2, 0.1])

    sky = create_mocksky(
        mock_key, mock_shape, mock_dist, (offset, fluctuations))

    comp_sky, reco_grid, data, data_grid = setup(mock_key)
    fig, axes = plt.subplots(1, 3)
    ax, ay, az = axes
    ax.imshow(sky, origin='lower')
    ay.imshow(comp_sky, origin='lower')
    az.imshow(data, origin='lower')
    plt.show()

    exit()

    CENTER = SkyCoord(0*u.rad, 0*u.rad)

    space = Space((1024, 1024), (DISTANCE, DISTANCE))
    mgrid = Grid(
        CENTER,
        shape=(1024, 1024),
        fov=(1024*DISTANCE*u.arcsec, 1024*DISTANCE*u.arcsec),
    )
    mxy = mgrid.rel_coords()
    xy = space.coords()
    assert np.allclose(xy, space.coords())

    rotation = 5 * u.deg
    smaller = 256
    rshape = (1024-smaller, 1024-smaller)
    downscale = 16
    dshape = [s//downscale for s in rshape]
    fov = [s*DISTANCE*u.arcsec for s in rshape]
    rgrid = Grid(CENTER, shape=rshape, fov=fov, rotation=rotation)
    dgrid = Grid(CENTER, shape=dshape, fov=fov, rotation=rotation)

    xy_rotated = rgrid.rel_coords()
    interpolation_points_old = (
        (xy_rotated - mgrid.rel_coords()[0, 0, 0]) / mgrid.distances[0].to(u.arcsec).value)
    interpolation_points = mgrid.wcs.index_from_wl(rgrid.wl_coords())[0]
    assert np.allclose(interpolation_points_old, interpolation_points)

    mask = np.full(rgrid.shape, True)
    nufft = build_nufft_integration(
        1, 1, interpolation_points[::-1, :, :][None], mask, sky.shape)
    sky_rot = nufft(sky)
    skysky = np.zeros(mask.shape)
    skysky[mask] = sky_rot
    data = downscale_sum(skysky, downscale)

    fig, ax = plt.subplots(1, 3)
    ax, ay, az = ax
    ax.imshow(sky, origin='lower')
    # plot_square(ax, arr[::-1, :])
    ay.imshow(skysky, origin='lower')
    az.imshow(data, origin='lower')
    arr = mgrid.wcs.index_from_wl(dgrid.wl_coords())[0]
    ax.scatter(*arr, color='orange')
    plt.show()
