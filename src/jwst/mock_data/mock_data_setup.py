import nifty8.re as jft

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u

from ..rotation_and_shift import build_nufft_rotation_and_shift
from ..reconstruction_grid import Grid


from jax import config
from jax import random
config.update('jax_platform_name', 'cpu')


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
):

    rota_grid = Grid(
        data_center, shape=rota_shape, fov=rota_fov, rotation=rotation)
    data_grid = Grid(
        data_center, shape=data_shape, fov=rota_fov, rotation=rotation)

    interpolation_points = mock_grid.wcs.index_from_wl(
        rota_grid.wl_coords())[0]

    nufft = build_nufft_rotation_and_shift(
        1, 1, interpolation_points[::-1, :, :], mock_grid.shape)
    rota_sky = nufft(mock_sky, None)

    downscale = [r//d for r, d in zip(rota_shape, data_shape)]
    assert downscale[0] == downscale[1]
    data = downscale_sum(rota_sky, downscale[0])

    return data_grid, data, rota_sky


def setup(
    mock_key,
    noise_key,
    **kwargs,
):

    noise_scale = kwargs['noise_scale']
    rotation = kwargs['rotations']
    reported_rotation = kwargs['reported_rotations']
    shift = kwargs['shifts']
    reported_shift = kwargs['reported_shifts']
    reco_shape = kwargs['reco_shape']
    sky_dict = kwargs['sky_dict']
    mock_distance = kwargs['mock_distance']
    mock_shape = kwargs['mock_shape']
    rota_shape = kwargs['rota_shape']
    data_shape = kwargs['data_shape']
    plot = kwargs.get('plot', False)

    cx, cy = 0, 0
    CENTER = SkyCoord(cx*u.rad, cy*u.rad)

    # True sky
    MOCK_SHAPE = (mock_shape,)*2
    MOCK_DIST = (mock_distance,)*2
    MOCK_FOV = [MOCK_SHAPE[ii]*(MOCK_DIST[ii]*u.arcsec) for ii in range(2)]
    mock_grid = Grid(CENTER, MOCK_SHAPE, MOCK_FOV)

    # Reco sky
    reco_shape = (reco_shape,)*2
    RECO_FOV = MOCK_FOV
    reco_grid = Grid(CENTER, reco_shape, RECO_FOV)
    comp_down = [r//d for r, d in zip(MOCK_SHAPE, reco_shape)]

    offset = sky_dict.get('offset')
    fluctuations = sky_dict.get('fluctuations')

    mock_sky = create_mocksky(
        mock_key, MOCK_SHAPE, MOCK_DIST, (offset, fluctuations))
    comparison_sky = downscale_sum(mock_sky, comp_down[0])

    # DATA SETUP
    ROTA_SHAPE = (rota_shape,)*2
    DATA_SHAPE = (data_shape,)*2
    rotation = rotation if isinstance(rotation, list) else [rotation]
    datas = {}
    for ii, (rot, repo_rot, shft, repo_shft) in enumerate(
            zip(rotation, reported_rotation, shift, reported_shift)):
        # Intermediate rotated sky
        ROTATION = rot*u.deg
        ROTA_FOV = [ROTA_SHAPE[ii]*(MOCK_DIST[ii]*u.arcsec) for ii in range(2)]

        # Data sky
        DATA_DIST = [r//d*MOCK_DIST[0] for r, d in zip(ROTA_SHAPE, DATA_SHAPE)]

        data_center = SkyCoord(cx*u.rad+(shft[0]*u.arcsec).to(u.rad),
                               cy*u.rad+(shft[1]*u.arcsec).to(u.rad))
        data_grid, data, rota_sky = create_data(
            mock_grid, mock_sky, data_center, ROTA_SHAPE, ROTA_FOV, DATA_SHAPE,
            ROTATION)

        # Create noise
        std = data.mean() * noise_scale
        data = data + random.normal(
            noise_key, data.shape, dtype=data.dtype) * std
        mask = np.full(data.shape, True)

        # UPDATE REPORTED CENTER, AND ROTATION.
        data_grid = Grid(
            SkyCoord((repo_shft[0]*u.arcsec).to(u.rad),
                     (repo_shft[1]*u.arcsec).to(u.rad)),
            shape=data_grid.shape,
            fov=data_grid.fov,
            rotation=repo_rot*u.deg)

        datas[f'd_{ii}'] = dict(data=data, mask=mask, std=std, grid=data_grid)

        if plot:
            arr = mock_grid.wcs.index_from_wl(data_grid.world_extrema()).T

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
                f'Comparison sky; sh={reco_shape[0]}, dist={comp_down[0]*MOCK_DIST[0]}')
            a11.set_title(f'data sky; sh={DATA_SHAPE[0]}, dist={DATA_DIST[0]}')

            for a, i in zip(ax.flatten(), ims):
                plt.colorbar(i, ax=a, shrink=0.7)
            plt.show()

    return comparison_sky, reco_grid, datas
