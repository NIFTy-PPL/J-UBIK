import yaml
import numpy as np

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

# import astropy
# import webbpsf
from astropy import units

from jwst_handling.integration_models import (
    build_sparse_integration,
    build_linear_integration,
    build_nufft_integration,
    build_integration_model,
    build_sparse_integration_model
)

from jwst_handling.data_model import JwstDataModel
from jwst_handling.sky_grid import ReconstructionGrid
from jwst_handling.masking import mask_index_centers_and_nan
from jwst_handling.config_handler import define_location, get_shape, get_fov

from sys import exit


import jax.numpy as jnp
from jax import random
from jax import config
config.update('jax_enable_x64', True)
config.update('jax_platform_name', 'cpu')


config_path = 'JWST_config.yaml'
config = yaml.load(open(config_path, 'r'), Loader=yaml.SafeLoader)
WORLD_LOCATION = define_location(config)
FOV = get_fov(config)
SHAPE = get_shape(config)


# defining the reconstruction grid
reco_grid = ReconstructionGrid(
    WORLD_LOCATION, SHAPE, (FOV.to(units.deg), FOV.to(units.deg)))


subsample = config['telescope']['integration_model']['subsample']
likelihoods = {}

for fltname, flt in config['files']['filter'].items():

    for ii, filepath in enumerate(flt):
        print(fltname, ii, filepath)
        jwst_data = JwstDataModel(filepath)

        # Find the sub-pixel centers for the interpolation integration
        subsample_centers = jwst_data.wl_subsample_centers(
            reco_grid.world_extrema(), subsample)
        index_subsample_centers = reco_grid.index_from_wl(
            subsample_centers)

        # Find the pixel edges for the sparse interpolation
        pix_center, (e00, e01, e10, e11) = jwst_data.wl_pixelcenter_and_edges(
            reco_grid.world_extrema())
        dpixcenter_in_rgrid = reco_grid.index_from_wl(pix_center)[0]
        index_edges = reco_grid.index_from_wl(
            [e00, e01, e11, e10])  # needs to be circular for sparse builder

        data = jwst_data.data_inside_extrema(reco_grid.world_extrema())
        std = jwst_data.std_inside_extrema(reco_grid.world_extrema())

        # define a mask
        mask = mask_index_centers_and_nan(
            dpixcenter_in_rgrid,
            data,
            reco_grid.shape
        )

        lh_name = f'{fltname}_{ii}'
        likelihoods[lh_name] = dict(
            mask=mask,
            index_edges=index_edges,
            index_subsample_centers=index_subsample_centers,
            data=data,
            std=std,
        )

for lh_name, lh in likelihoods.items():
    ind_grid = reco_grid.index_grid(config['grid']['padding_ratio'])
    lh['sparse_matrix'] = build_sparse_integration(
        ind_grid, lh['index_edges'], lh['mask'])

    lh['interpolation'] = build_linear_integration(
        lh['index_subsample_centers'], mask=lh['mask'], order=1)

    lh['updating_interpolation'] = build_linear_integration(
        lh['index_subsample_centers'], mask=lh['mask'], order=1, updating=True)

    lh['nufft_interpolation'] = build_nufft_integration(
        lh['index_subsample_centers'], mask=lh['mask'],
        shape=np.array(ind_grid[0].shape))


if __name__ == '__main__':
    import jubik0 as ju
    import nifty8.re as jft

    def build_shift_model(key, mean_sigma):
        from charm_lensing.models.parametric_models.parametric_prior import (
            build_prior_operator)
        distribution_model_key = ('normal', *mean_sigma)
        shape = (2,)

        shift_model = build_prior_operator(key, distribution_model_key, shape)
        domain = {key: jft.ShapeWithDtype((shape))}
        return jft.Model(shift_model, domain=domain)

    mock = False
    plots = True

    integration_model = config['telescope']['integration_model']['model']
    mean, sigma = (config['telescope']['integration_model']['mean'],
                   config['telescope']['integration_model']['sigma'])
    mean_sigma = (mean, sigma)

    key = random.PRNGKey(42)
    key, subkey, sampling_key, mini_par_key = random.split(key, 4)

    sky_dict = ju.create_sky_model_from_config(config_path)
    sky = sky_dict['sky_full']
    # sky = sky_dict['sky']

    likes = []
    models = []
    for lh_key, lh in likelihoods.items():

        lh['sparse_model'] = build_sparse_integration_model(
            lh['sparse_matrix'], sky)

        lh['interpolation_model'] = build_integration_model(
            lh['interpolation'], sky)

        shift = build_shift_model(lh_key+'shift', mean_sigma)
        lh['shift_model'] = shift
        lh['updating_interpolation_model'] = build_integration_model(
            lh['updating_interpolation'], sky, shift)

        lh['nufft_model'] = build_integration_model(
            lh['nufft_interpolation'], sky)

        if mock:
            mock_sky = sky(jft.random_like(sampling_key, sky.domain))
            data = lh['sparse_matrix'] @ mock_sky.reshape(-1)
            scale = 0.1*data.mean()
            data += np.random.normal(scale=scale, size=data.shape)
            std = np.full(data.shape, scale)

            plot_data = np.zeros_like(lh['data'])
            mask = lh['mask']
            plot_data[mask] = data

        else:
            mask = lh['mask']
            data = jnp.array(lh['data'][mask], dtype=jnp.float64)
            std = jnp.array(lh['std'][mask], dtype=jnp.float64)
            plot_data = lh['data']

            pos = jft.random_like(sampling_key, sky.domain)
            tmp = np.zeros_like(plot_data)
            sparse_model = lh['sparse_model']
            tmp[mask] = sparse_model(pos)

            tmp2 = np.zeros_like(plot_data)
            interpolation_model = lh['interpolation_model']
            tmp2[mask] = lh['interpolation_model'](pos)

            tmp3 = np.zeros_like(plot_data)
            pos3 = jft.random_like(
                mini_par_key, lh['updating_interpolation_model'].domain)
            shif = lh['shift_model'](pos3)
            tmp3[mask] = lh['updating_interpolation_model'](pos3)

            tmp4 = np.zeros_like(plot_data)
            tmp4[mask] = lh['nufft_model'](pos)

            log_min, log_max = 1.0, tmp.max()
            if plots:
                fig, axes = plt.subplots(3, 3, sharex=True, sharey=True)
                axes = axes.flatten()
                ims = []
                ims.append(axes[0].imshow(
                    plot_data, origin='lower', norm=LogNorm()))
                axes[0].set_title('data')
                ims.append(axes[1].imshow(plot_data-tmp,
                           origin='lower', norm=LogNorm()))
                axes[1].set_title('data - sparse')
                ims.append(axes[2].imshow(sky_dict['sky'](pos),
                           origin='lower', norm=LogNorm(vmin=log_min, vmax=log_max)))
                axes[2].set_title('data - finufft')
                ims.append(axes[3].imshow(
                    tmp, origin='lower',
                    norm=LogNorm(vmin=log_min, vmax=log_max)))
                axes[3].set_title('sparse')
                ims.append(axes[4].imshow(
                    tmp2, origin='lower', norm=LogNorm(vmin=log_min, vmax=log_max)))
                axes[4].set_title('interpolation')
                ims.append(axes[5].imshow(
                    tmp4, origin='lower', norm=LogNorm(vmin=log_min, vmax=log_max)))
                axes[5].set_title('nufft')
                ims.append(axes[6].imshow((tmp-tmp2)/tmp,
                           origin='lower', cmap='RdBu_r', vmin=-1, vmax=1))
                axes[6].set_title('sparse - interpolation')
                ims.append(axes[7].imshow((tmp-tmp3)/tmp,
                           origin='lower', cmap='RdBu_r', vmin=-1, vmax=1))
                axes[7].set_title('sparse - update_interpolation')
                ims.append(axes[8].imshow((tmp-tmp4)/tmp,
                           origin='lower', cmap='RdBu_r', vmin=-1, vmax=1))
                axes[8].set_title('sparse-nufft')
                for im, ax in zip(ims, axes):
                    fig.colorbar(im, ax=ax, shrink=0.7)
                plt.show()

        like = ju.library.likelihood.build_gaussian_likelihood(data, std)
        if integration_model in ['sparse']:
            like = like.amend(sparse_model, domain=sparse_model.domain)
        elif integration_model in ['interpolation']:
            like = like.amend(
                lh['interpolation_model'], domain=lh['interpolation_model'].domain)
        elif integration_model in ['updating_interpolation']:
            like = like.amend(
                lh['updating_interpolation_model'], domain=lh['updating_interpolation_model'].domain)
        elif integration_model in ['nufft']:
            like = like.amend(
                lh['nufft_model'], domain=lh['nufft_model'].domain)

        likes.append(like)

    from functools import reduce
    like = reduce(lambda a, b: a + b, likes)

    cfg = ju.get_config(config_path)
    file_info = cfg['files']
    minimization_config = cfg['minimization']
    kl_solver_kwargs = minimization_config.pop('kl_kwargs')

    sky_dict.pop('target')
    sky_dict.pop('pspec')

    def plot(s, x):
        ju.plot_sample_and_stats(file_info["res_dir"],
                                 sky_dict,
                                 s,
                                 log_scale=True,
                                 relative_std=True,
                                 iteration=x.nit)

        from os.path import join
        from os import makedirs
        out_dir = join(file_info["res_dir"], 'residuals')
        makedirs(out_dir, exist_ok=True)

        fig, axes = plt.subplots(
            len(likelihoods.values()), 3, sharex=True, sharey=True,
            figsize=(9, 3 * len(likelihoods.values())),
            dpi=300)
        if len(likelihoods.values()) == 1:
            axes = np.array([axes])
        for ii, lh in enumerate(likelihoods.values()):

            if integration_model in ['sparse']:
                model = lh['sparse_model']
            elif integration_model in ['interpolation']:
                model = lh['interpolation_model']
            elif integration_model in ['updating_interpolation']:
                model = lh['updating_interpolation_model']
            elif integration_model in ['nufft']:
                model = lh['nufft_model']

            plot_data = lh['data']
            mask = lh['mask']

            arr = np.zeros_like(plot_data)
            arr[mask] = jft.mean([model(si) for si in s])
            im0 = axes[ii, 0].imshow(plot_data, origin='lower', norm=LogNorm())
            im1 = axes[ii, 1].imshow(arr, origin='lower', norm=LogNorm())
            im2 = axes[ii, 2].imshow((plot_data - arr)/lh['std'], origin='lower',
                                     cmap='RdBu_r', vmax=3, vmin=-3)
            for ax, im in zip(axes[ii], [im0, im1, im2]):
                fig.colorbar(im, ax=ax, shrink=0.7)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/residuals_{x.nit}.png')

        if mock:
            sky_mean = jft.mean([sky(si) for si in s])
            fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
            im0 = axes[0].imshow(mock_sky, origin='lower')
            im1 = axes[1].imshow(sky_mean, origin='lower')
            im2 = axes[2].imshow((mock_sky - sky_mean),
                                 origin='lower', cmap='RdBu_r')
            for ax, im in zip(axes, [im0, im1, im2]):
                fig.colorbar(im, ax=ax, shrink=0.7)
            plt.show()

    pos_init = 0.1 * jft.Vector(jft.random_like(subkey, like.domain))
    samples, state = jft.optimize_kl(
        like,
        pos_init,
        key=key,
        kl_kwargs=kl_solver_kwargs,
        callback=plot,
        odir=file_info["res_dir"],
        **minimization_config)

    pos = jft.mean(samples)
