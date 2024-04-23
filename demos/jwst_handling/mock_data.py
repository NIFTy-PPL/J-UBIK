import nifty8.re as jft
import jax.numpy as jnp
import numpy as np

import jax
jax.config.update('jax_platform_name', 'cpu')


def point_in_polygon(point, polygon):
    """Determine if the point (x, y) is inside the given polygon.
    polygon is a list of tuples [(x1, y1), (x2, y2), ..., (xn, yn)]"""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n+1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def pixels_in_rectangle(corners, shape):
    """Get a boolean numpy array where each element indicates whether the center of the pixel at that index is inside the rectangle."""
    # Convert corners to a list of tuples
    polygon = [(corners[i][0], corners[i][1]) for i in range(corners.shape[0])]

    pixel_map = np.zeros(shape, dtype=bool)

    min_x = np.floor(np.min(corners[:, 0]))
    max_x = np.ceil(np.max(corners[:, 0]))
    min_y = np.floor(np.min(corners[:, 1]))
    max_y = np.ceil(np.max(corners[:, 1]))

    start_x = max(0, int(min_x))
    end_x = min(shape[1], int(max_x) + 1)
    start_y = max(0, int(min_y))
    end_y = min(shape[0], int(max_y) + 1)

    for x in range(start_x, end_x):
        for y in range(start_y, end_y):
            if point_in_polygon((x + 0.5, y + 0.5), polygon):
                pixel_map[x, y] = True

    return pixel_map


def smallest_enclosing_mask(pixel_map):
    """Finds the edge points of the largest and smallest `True` pixels in specified sections of the pixel_map."""
    assert pixel_map.shape[0] == pixel_map.shape[1]

    mask = np.zeros_like(pixel_map)

    shape = pixel_map.shape[0]
    for ii in range(shape):
        if np.all(pixel_map[ii:shape-ii, ii:shape-ii]):
            mask[ii:shape-ii, ii:shape-ii] = True
            return ii, mask

    return ii, mask


def build_plot(datas, data_models, masks, plot_sky, sky_model, res_dir, vals_calc_mask):
    from charm_lensing.analysis_tools import source_distortion_ratio
    from scipy.stats import wasserstein_distance
    from charm_lensing.plotting import display_text
    from charm_lensing.analysis_tools import wmse, redchi2
    from nifty8.re.caramel import power_analyze
    from os.path import join
    from os import makedirs

    out_dir = join(res_dir, 'residuals')
    makedirs(out_dir, exist_ok=True)

    def get_power(field):
        return power_analyze(
            np.fft.fft2(field[smallest_mask].reshape((reshape,)*2)),
            (1.0,)*2  # FAKE DISTANCES ::FIXME::
        )

    def cross_correlation(input, recon):
        return np.fft.ifft2(
            np.fft.fft2(input).conj() * np.fft.fft2(recon)
        ).real.max()

    eval_comp_sky = np.zeros_like(plot_sky)
    eval_self_sky = np.zeros_like(plot_sky)

    ii, smallest_mask = smallest_enclosing_mask(vals_calc_mask)
    reshape = vals_calc_mask.shape[0] - 2*ii
    true_power_spectrum = get_power(plot_sky)

    def plot_pspec(samples, x):
        YLIMS = (1e2, 1e11)

        skys = [sky_model(si) for si in samples]

        pws = [get_power(sky) for sky in skys]
        pw = get_power(jft.mean(skys))

        fig, ax = plt.subplots(1, 1, figsize=(9, 3), dpi=300)
        ax.plot(pw, label='reco', color='blue', linewidth=0.5)
        for pw in pws:
            ax.plot(pw, color='blue', alpha=0.5, linewidth=0.3)
        ax.plot(true_power_spectrum, label='true',
                color='orange', linewidth=0.5)
        plt.ylim(YLIMS)
        ax.loglog()
        ax.legend()
        fig.savefig(join(out_dir, f'pspec_{x.nit:02d}.png'), dpi=300)
        plt.close()

    def plot(samples, x):
        plot_pspec(samples, x)

        sky = jft.mean([sky_model(si) for si in samples])

        eval_comp_sky[vals_calc_mask] = plot_sky[vals_calc_mask]
        eval_self_sky[vals_calc_mask] = sky[vals_calc_mask]

        vals = dict(
            sdr=source_distortion_ratio(eval_comp_sky, eval_self_sky),
            wd=wasserstein_distance(eval_comp_sky.reshape(-1),
                                    eval_self_sky.reshape(-1)),
            # cc=cross_correlation(eval_comp_sky, eval_self_sky),
        )

        ylen = 1+len(datas)
        fig, axes = plt.subplots(ylen, 3, figsize=(9, 3*ylen), dpi=300)
        ims = []
        for ii, (d, dm, mask) in enumerate(zip(datas, data_models, masks)):
            model_data = []
            for si in samples:
                tmp = np.zeros_like(data)
                tmp[mask] = dm(si)
                model_data.append(tmp)

            mod_mean = jft.mean(model_data)
            redchi_mean, redchi2_std = jft.mean_and_std(
                [redchi2(d, m, std, d.size) for m in model_data])

            axes[ii, 0].set_title('Data')
            ims.append(axes[ii, 0].imshow(d, origin='lower'))
            axes[ii, 1].set_title('Data model')
            ims.append(axes[ii, 1].imshow(mod_mean, origin='lower'))
            axes[ii, 2].set_title('Data residual')
            ims.append(axes[ii, 2].imshow((d - mod_mean)/std, origin='lower',
                                          vmin=-3, vmax=3, cmap='RdBu_r'))
            chi = '\n'.join((
                f'MSE/var: {wmse(d, mod_mean, std):.2f}',
                f'redChi2: {redchi_mean:.2f} +/- {redchi2_std:.2f}',
            ))

            display_text(axes[ii, 2], chi)

        axes[ii+1, 0].set_title('Sky')
        ims.append(axes[ii+1, 0].imshow(plot_sky, origin='lower'))
        axes[ii+1, 1].set_title('Sky model')
        ims.append(axes[ii+1, 1].imshow(sky, origin='lower'))
        axes[ii+1, 2].set_title('Sky residual')
        ims.append(axes[ii+1, 2].imshow(
            (plot_sky - sky)/plot_sky, origin='lower',
            vmin=-0.3, vmax=0.3, cmap='RdBu_r'))

        axes[ii+1, 0].contour(vals_calc_mask.T, levels=1, colors='orange')
        axes[ii+1, 1].contour(vals_calc_mask.T, levels=1, colors='orange')
        axes[ii+1, 2].contour(vals_calc_mask.T, levels=1, colors='orange')

        ss = '\n'.join(
            [f'{k}: {v:.3f}' if k != 'cc' else f'{k}: {v:e}' for k, v in vals.items()])
        display_text(axes[ii+1, 2], ss)
        for ax, im in zip(axes.flatten(), ims):
            fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        fig.savefig(join(out_dir, f'{x.nit:02d}.png'), dpi=300)
        plt.close()

    return plot


def find_corners(xy_positions):
    xy_positions = np.array(xy_positions)

    maxx, maxy = np.argmax(xy_positions, axis=1)
    minx, miny = np.argmin(xy_positions, axis=1)

    square_corners = np.array([
        xy_positions[:, maxx],
        xy_positions[:, maxy],
        xy_positions[:, minx],
        xy_positions[:, miny],
    ])
    return square_corners


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


def setup(mock_key, rotation, reco_shape, rota_shape=768, data_shape=48, plot=False):
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
    for ii, rot in enumerate(rotation):
        # Intermediate rotated sky
        ROTATION = rot*u.deg
        ROTA_FOV = [ROTA_SHAPE[ii]*(MOCK_DIST[ii]*u.arcsec) for ii in range(2)]

        # Data sky
        DATA_DIST = [r//d*MOCK_DIST[0] for r, d in zip(ROTA_SHAPE, DATA_SHAPE)]

        data_grid, data, rota_sky = create_data(
            CENTER, mock_grid, mock_sky, ROTA_SHAPE, ROTA_FOV, DATA_SHAPE,
            ROTATION, full_info=True)
        datas[f'd_{ii}'] = dict(data=data, grid=data_grid)

        if plot:
            import matplotlib.pyplot as plt
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


def build_sky_model(shape, dist):

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

    reco_diffuse = cfm.finalize()
    # sky_model = jft.Model(
    #     lambda x: jnp.exp(reco_diffuse(x)[:reco_shape[0], :reco_shape[1]]),
    #     domain=reco_diffuse.domain)

    sky_model_full = jft.Model(
        lambda x: jnp.exp(reco_diffuse(x)),
        domain=reco_diffuse.domain)

    return sky_model_full


if __name__ == '__main__':
    from jax import random
    import matplotlib.pyplot as plt
    from jwst_handling.integration_models import (
        build_sparse_integration, build_sparse_integration_model,
        build_linear_integration, build_integration_model,
        build_nufft_integration)
    from astropy import units as u
    import jubik0 as ju

    # Reconstruction Setup
    SUBSAMPLE = 7
    NOISE_SCALE = 0.01
    MODEL = 'linear'
    RSHAPE = 128
    ROTATION_0 = 2
    ROTATION_1 = 23
    PLOT = False

    key = random.PRNGKey(87)
    key, mock_key, noise_key, rec_key = random.split(key, 4)
    comp_sky, reco_grid, data_set = setup(
        mock_key,
        rotation=[ROTATION_0, ROTATION_1],
        reco_shape=RSHAPE,
        plot=PLOT)

    sky_model = build_sky_model(
        reco_grid.shape, [d.to(u.arcsec).value for d in reco_grid.distances])

    if False:
        key, check_key = random.split(key)
        m = sky_model(jft.random_like(check_key, sky_model.domain))
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

    evaluation_mask = np.full(reco_grid.shape, False)
    datas = []
    models = []
    masks = []
    likelihoods = []
    for data_key in data_set.keys():
        data, data_grid = data_set[data_key]['data'], data_set[data_key]['grid']

        # For interpolation
        wl_data_subsample_centers = data_grid.wcs.wl_subsample_centers(
            data_grid.world_extrema, SUBSAMPLE)
        px_reco_subsample_centers = reco_grid.wcs.index_from_wl(
            wl_data_subsample_centers)
        px_reco_subsample_centers = px_reco_subsample_centers[:, ::-1, :, :]

        # For sparse
        wl_data_centers, (e00, e01, e10, e11) = data_grid.wcs.wl_pixelcenter_and_edges(
            data_grid.world_extrema)
        px_reco_datapix_cntr = reco_grid.wcs.index_from_wl(wl_data_centers)[0]
        px_reco_index_edges = reco_grid.wcs.index_from_wl(
            [e00, e01, e11, e10])  # needs to be circular for sparse builder

        corners = find_corners(px_reco_datapix_cntr.reshape(2, -1))
        tmp_mask = pixels_in_rectangle(corners, comp_sky.shape)
        evaluation_mask += tmp_mask

        if PLOT:
            fig, axes = plt.subplots(1, 3)
            ax, ay = axes
            ax.imshow(comp_sky)
            ax.scatter(*corners.T[::-1], color='orange')
            ay.imshow(tmp_mask)
            ay.scatter(*corners.T[::-1], color='orange')
            plt.show()

        std = data.mean() * NOISE_SCALE
        d = data + random.normal(noise_key, data.shape, dtype=data.dtype) * std
        likelihood_func = ju.library.likelihood.build_gaussian_likelihood(
            d.reshape(-1), float(std))

        mask = np.full(data.shape, True)
        sky_dvol = reco_grid.dvol
        sub_dvol = data_grid.dvol / (SUBSAMPLE**2)

        if MODEL == 'sparse':
            sparse_matrix = build_sparse_integration(
                reco_grid.index_grid(),
                px_reco_index_edges,
                mask)
            # FIXME: This is not optimal; better distribute the model to the operator
            model = build_sparse_integration_model(sparse_matrix, sky_model)
            res_dir = f'results/mock_rotation/c{ROTATION_0}_{ROTATION_1}/{RSHAPE}_sparse'

        elif MODEL == 'linear':
            linear = build_linear_integration(
                sky_dvol,
                sub_dvol,
                px_reco_subsample_centers,
                mask,
                order=1,
                updating=False)
            model = build_integration_model(linear, sky_model)
            res_dir = f'results/mock_rotation/c{ROTATION_0}_{ROTATION_1}/{RSHAPE}_linear_sub{SUBSAMPLE}'

        elif MODEL == 'nufft':
            nufft = build_nufft_integration(
                sky_dvol=sky_dvol,
                sub_dvol=sub_dvol,
                subsample_centers=px_reco_subsample_centers,
                mask=mask,
                sky_shape=reco_grid.shape,
            )
            model = build_integration_model(nufft, sky_model)
            res_dir = f'results/mock_rotation/c{ROTATION_0}_{ROTATION_1}/{RSHAPE}_nufft'

        masks.append(mask)
        datas.append(d)
        models.append(model)
        likelihoods.append(likelihood_func.amend(model, domain=model.domain))

    from functools import reduce
    like = reduce(lambda a, b: a + b, likelihoods)

    plot = build_plot(
        datas, models, masks, comp_sky, sky_model, res_dir, evaluation_mask)

    pos_init = 0.1 * jft.Vector(jft.random_like(rec_key, like.domain))

    cfg = ju.get_config('./JWST_config.yaml')
    minimization_config = cfg['minimization']
    kl_solver_kwargs = minimization_config.pop('kl_kwargs')
    minimization_config['n_total_iterations'] = 25
    # minimization_config['resume'] = True
    minimization_config['n_samples'] = lambda it: 4 if it < 10 else 10

    print(f'Results: {res_dir}')
    samples, state = jft.optimize_kl(
        like,
        pos_init,
        key=rec_key,
        kl_kwargs=kl_solver_kwargs,
        callback=plot,
        odir=res_dir,
        **minimization_config)

    plot = build_plot(
        datas, models, masks, comp_sky, sky_model, res_dir, evaluation_mask)
    plot(samples, state)
