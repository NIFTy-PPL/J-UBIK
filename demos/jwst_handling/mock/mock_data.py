import nifty8.re as jft
import jax.numpy as jnp
import numpy as np

import jax
jax.config.update('jax_platform_name', 'cpu')


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


def create_data(key, shapes, mock_dist, model_setup, show=True):
    offset, fluctuations = model_setup
    mock_shape, reco_shape,  data_shape = shapes

    cfm = jft.CorrelatedFieldMaker(prefix='mock')
    cfm.set_amplitude_total_offset(**offset)
    cfm.add_fluctuations(
        mock_shape, mock_dist, **fluctuations, non_parametric_kind='power')
    mock_diffuse = cfm.finalize()
    mock_sky = jnp.exp(mock_diffuse(jft.random_like(key, mock_diffuse.domain)))

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


def plot_surrounding_square(ax, xy_positions, color='r'):
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from jax import random
    from charm_lensing.spaces import Space
    from charm_lensing.models.parametric_models.rotation import rotation
    from jwst_handling.integration_models import (
        build_nufft_integration
    )

    key = random.PRNGKey(87)
    RSHAPE = 256
    key, mock_key, noise_key, rec_key = random.split(key, 4)
    mock_shape, mock_dist = (1024, 1024), (0.5, 0.5)
    reco_shape, reco_dist = (RSHAPE,)*2, (0.5*1024.0/RSHAPE,)*2
    data_shape, data_dist = (64, 64), (8.0, 8.0)

    offset = dict(offset_mean=0.1, offset_std=[0.1, 0.05])
    fluctuations = dict(fluctuations=[0.3, 0.03],
                        loglogavgslope=[-3., 1.],
                        flexibility=[0.8, 0.1],
                        asperity=[0.2, 0.1])

    sky, *_ = create_data(
        mock_key,
        (mock_shape, reco_shape, data_shape),
        mock_dist,
        (offset, fluctuations),
        show=False
    )

    space = Space((1024, 1024), (0.5, 0.5))
    xy = space.coords()

    deg_rot = 30
    rots, crops = [], []
    for ii in range(3):
        deg_rot += 10
        xy_rotated = jnp.array(rotation(xy, deg_rot*jnp.pi/180))
        mask = ~(
            ((xy_rotated[0] < space.extent[0]) + (xy_rotated[0] > space.extent[1])) +
            ((xy_rotated[1] < space.extent[2]) +
             (xy_rotated[1] > space.extent[3]))
        )

        interpolation_points = (xy_rotated - xy[0, 0, 0]) / space.distances[0]
        interpolation_points = jnp.array(
            (interpolation_points[1], interpolation_points[0]))[None]

        nufft = build_nufft_integration(
            1, 1, interpolation_points, mask, sky.shape)
        sky_rot = nufft(sky)

        down = 128+64

        rotated = np.zeros_like(sky)
        rotated[~mask] = np.nan
        rotated[mask] = sky_rot
        cropped_xy = xy_rotated[:, down:-down, down:-down]
        rots.append(rotated)
        crops.append(cropped_xy)

    fig, axes = plt.subplots(3, 3)
    # axes = axes.flatten()
    for ii, ax in enumerate(axes):
        cropped_xy = crops[ii]
        rotated = rots[ii]
        ax[0].imshow(sky, origin='lower', extent=space.extent)
        plot_surrounding_square(
            ax[0], cropped_xy.reshape(2, -1), color='orange')
        ax[1].imshow(rotated[down: -down, down: -down], origin='lower')
        dsd = downscale_sum(rotated[down: -down, down: -down], 8)
        ax[2].imshow(dsd, origin='lower')
    plt.show()
