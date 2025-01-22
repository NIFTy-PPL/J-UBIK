import nifty8.re as jft

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u

from ..rotation_and_shift import build_nufft_rotation_and_shift
from ..rotation_and_shift import build_linear_rotation_and_shift
from ..grid import Grid


from jax import random


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


def plot_setup(
    mock_sky, mock_grid,
    rota_sky,
    comparison_sky, comp_down,
    data, data_grid,
):

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
        f'Underlying sky; sh={mock_grid.shape[0]}, dist={mock_grid.distances[0].to(u.arcsec).value:.2f}')
    a01.set_title(
        f'Cut out; sh={rota_sky.shape[0]}, dist={mock_grid.distances[0].to(u.arcsec).value:.2f}')
    a10.set_title(
        f'Comparison sky; sh={comparison_sky.shape[0]}, dist={comp_down[0]*mock_grid.distances[0].to(u.arcsec).value:.2f}')
    a11.set_title(
        f'data sky; sh={data_grid.shape[0]}, dist={data_grid.distances[0].to(u.arcsec).value:.2f}')

    for a, i in zip(ax.flatten(), ims):
        plt.colorbar(i, ax=a, shrink=0.7)
    plt.show()


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
    rotation_model_key,
):

    rota_grid = Grid(
        data_center, shape=rota_shape, fov=rota_fov, rotation=rotation)
    data_grid = Grid(
        data_center, shape=data_shape, fov=rota_fov, rotation=rotation)

    interpolation_points = mock_grid.wcs.index_from_wl(
        rota_grid.wl_coords())[0]

    if rotation_model_key == 'nufft':
        print('data created with nufft rotation and shift')
        nufft = build_nufft_rotation_and_shift(
            1, 1, mock_grid.shape, interpolation_points.shape[1:])
        rota_sky = nufft(mock_sky, interpolation_points)

    elif rotation_model_key == 'linear':
        print('data created with linear rotation and shift')
        linear = build_linear_rotation_and_shift(1, 1, order=1)
        rota_sky = linear(mock_sky, interpolation_points)

    else:
        msg = f'{rotation_model_key} model does not exist for data creation'
        raise NotImplementedError(msg)

    downscale = [r//d for r, d in zip(rota_shape, data_shape)]
    assert downscale[0] == downscale[1]
    data = downscale_sum(rota_sky, downscale[0])

    return data_grid, data, rota_sky


def mock_setup(
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
    data_creation_rotation_model_key = kwargs.get('rotation', 'nufft')

    cx, cy = 0, 0
    center = SkyCoord(cx*u.rad, cy*u.rad)

    # True sky
    mock_shape = (mock_shape,)*2
    mock_dist = (mock_distance,)*2
    mock_fov = [mock_shape[ii]*(mock_dist[ii]*u.arcsec) for ii in range(2)]
    mock_grid = Grid(center, mock_shape, mock_fov)

    # Reco sky
    reco_shape = (reco_shape,)*2
    reco_grid = Grid(center, reco_shape, mock_fov)
    comp_down = [r//d for r, d in zip(mock_shape, reco_shape)]

    offset = sky_dict.get('offset')
    fluctuations = sky_dict.get('fluctuations')

    mock_sky = create_mocksky(
        mock_key, mock_shape, mock_dist, (offset, fluctuations))
    comparison_sky = downscale_sum(mock_sky, comp_down[0])

    # DATA SETUP
    rota_shape = (rota_shape,)*2
    data_shape = (data_shape,)*2
    rotation = rotation if isinstance(rotation, list) else [rotation]
    datas = {}
    for ii, (rot, repo_rot, shft, repo_shft) in enumerate(
            zip(rotation, reported_rotation, shift, reported_shift)):

        # Data sky
        data_grid, data_no_noise, rota_sky = create_data(
            mock_grid=mock_grid,
            mock_sky=mock_sky,
            data_center=SkyCoord(
                cx*u.rad+(shft[0]*u.arcsec).to(u.rad),
                cy*u.rad+(shft[1]*u.arcsec).to(u.rad)
            ),
            rota_shape=rota_shape,
            rota_fov=[rota_shape[ii]*(mock_dist[ii]*u.arcsec)
                      for ii in range(2)],
            data_shape=data_shape,
            rotation=rot*u.deg,
            rotation_model_key=data_creation_rotation_model_key,
        )

        # Create noise
        key, noise_key = random.split(noise_key)
        std = data_no_noise.mean() * noise_scale
        data = data_no_noise + random.normal(
            noise_key, data_no_noise.shape, dtype=data_no_noise.dtype) * std
        mask = np.full(data.shape, True)

        # Save reported center, shift, and rotation
        data_grid = Grid(
            SkyCoord((repo_shft[0]*u.arcsec).to(u.rad),
                     (repo_shft[1]*u.arcsec).to(u.rad)),
            shape=data_grid.shape,
            fov=data_grid.fov,
            rotation=repo_rot*u.deg)

        datas[f'd_{ii}'] = dict(
            data=data,
            mask=mask,
            std=std,
            grid=data_grid,
            shift=dict(true=shft, reported=repo_shft),
            rotation=dict(true=rot, reported=repo_rot),
            rota_sky=rota_sky,
            # mock_sky=mock_sky,
            data_no_noise=data_no_noise,
        )

        if plot:
            plot_setup(
                mock_sky,
                mock_grid,
                rota_sky,
                comparison_sky,
                comp_down,
                data,
                data_grid,
            )
    return comparison_sky, reco_grid, datas
