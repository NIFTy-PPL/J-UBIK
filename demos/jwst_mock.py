from sys import exit
import nifty8.re as jft

import numpy as np
import matplotlib.pyplot as plt

from jwst_handling.mock_data import setup

from jwst_handling.integration_models import (
    build_sparse_integration,
    build_sparse_integration_model,
    build_linear_integration,
    build_nufft_integration,
    build_integration_model
)
import jubik0 as ju

import jax.numpy as jnp
from jax import config, random
config.update('jax_enable_x64', True)
config.update('jax_platform_name', 'cpu')

SHOW_DATA = False
PRIOR_SAMPLE = False
PADDING = 1.5
STD_FACTOR = 0.02
MODEL = 'sparse'  # 'sparse', 'linear', 'nufft'
SUBSAMPLE = 5
MSHAPE = 1024
RSHAPE = 128
DSHAPE = 64

subsample = SUBSAMPLE

mock_shape, mock_dist = (MSHAPE,)*2, (0.5,)*2
reco_shape, reco_dist = (RSHAPE,)*2, (0.5*1024.0/RSHAPE,)*2
data_shape, data_dist = (DSHAPE,)*2, (0.5*1024.0/DSHAPE,)*2

key = random.PRNGKey(42)
key, mock_key, noise_key, rec_key = random.split(key, 4)

offset = dict(offset_mean=0.1, offset_std=[0.1, 0.05])
fluctuations = dict(fluctuations=[0.3, 0.03],
                    loglogavgslope=[-3., 1.],
                    flexibility=[0.8, 0.1],
                    asperity=[0.2, 0.1])

comp_sky, reco_grid, data_set = setup(
    mock_key, rotation=[0],
    reco_shape=RSHAPE,
    rota_shape=MSHAPE,
    data_shape=DSHAPE,
    plot=False)
data = data_set['d_0']['data']
data_grid = data_set['d_0']['grid']
mask = np.full(data.shape, True)


offset = dict(offset_mean=3.7, offset_std=[0.1, 0.05])
fluctuations = dict(fluctuations=[0.7, 0.03],
                    loglogavgslope=[-4.8, 1.],
                    flexibility=[0.8, 0.1],
                    asperity=[0.2, 0.1])
cfm = jft.CorrelatedFieldMaker(prefix='reco')
cfm.set_amplitude_total_offset(**offset)
cfm.add_fluctuations(
    [int(shp * PADDING) for shp in reco_shape],
    mock_dist, **fluctuations, non_parametric_kind='power')
reco_diffuse = cfm.finalize()
sky_model = jft.Model(
    lambda x: jnp.exp(reco_diffuse(x)[:reco_shape[0], :reco_shape[1]]),
    domain=reco_diffuse.domain)
sky_model_full = jft.Model(
    lambda x: jnp.exp(reco_diffuse(x)),
    domain=reco_diffuse.domain)

if PRIOR_SAMPLE:
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


def get_sparse_model(mask):
    wl_data_centers, (e00, e01, e10, e11) = data_grid.wcs.wl_pixelcenter_and_edges(
        data_grid.world_extrema)
    px_reco_index_edges = reco_grid.wcs.index_from_wl(
        [e00, e01, e11, e10])  # needs to be circular for sparse builder

    sparse_matrix = build_sparse_integration(
        reco_grid.index_grid(),
        px_reco_index_edges,
        mask)

    return build_sparse_integration_model(sparse_matrix, sky_model)


def get_subsample_centers(subsample):
    factor = [int(dd/rd) for dd, rd in zip(data_dist, reco_dist)]
    pix_center = np.array(np.meshgrid(
        *[np.arange(ds)*f for ds, f in zip(data_shape, factor)]
    )) + np.array([1/f for f in factor])[..., None, None]
    pix_center = pix_center[::-1, :, :]

    d = pix_center[0, 1, 0] - pix_center[0, 0, 0]
    ps = (np.arange(0.5/subsample, 1, 1/subsample) - 0.5) * d
    ms = np.vstack(np.array(np.meshgrid(ps, ps)).T)
    ms = ms[:, ::-1]
    return ms[:, :, None, None] + pix_center


def get_linear_model(subsample, mask):
    subsample_centers = get_subsample_centers(subsample)

    sky_dvol = np.prod(reco_dist)
    sub_dvol = np.prod((data_dist[0] / subsample, data_dist[1] / subsample))

    linear = build_linear_integration(
        sky_dvol,
        sub_dvol,
        subsample_centers,
        mask,
        order=1,
        updating=False)

    return build_integration_model(linear, sky_model_full)


def get_nufft_model(subsample, mask):
    subsample_centers = get_subsample_centers(subsample)
    sky_dvol = np.prod(reco_dist)
    sub_dvol = np.prod((data_dist[0] / subsample, data_dist[1] / subsample))

    nufft = build_nufft_integration(
        sky_dvol,
        sub_dvol,
        subsample_centers,
        mask,
        sky_model_full.target.shape)
    return build_integration_model(nufft, sky_model_full)


std = STD_FACTOR*data.mean()
d = data + random.normal(noise_key, data.shape, dtype=data.dtype) * std
if MODEL == 'sparse':
    model = get_sparse_model_new(mask)

    res_dir = f'results/mock_integration/{RSHAPE}_sparse'
elif MODEL == 'linear':
    model = get_linear_model(subsample=SUBSAMPLE, mask=mask)
    res_dir = f'results/mock_integration/{RSHAPE}_linear{SUBSAMPLE}'
elif MODEL == 'nufft':
    model = get_nufft_model(subsample=SUBSAMPLE, mask=mask)
    res_dir = f'results/mock_integration/{RSHAPE}_nufft{SUBSAMPLE}'

res_dir += '_new'
print('*'*80)
print(f'\nResults: {res_dir}\n')
print('*'*80)

like = ju.library.likelihood.build_gaussian_likelihood(
    d.reshape(-1), float(std))
like = like.amend(model, domain=model.domain)


def build_plot(plot_data, plot_sky, mask, data_model, sky_model, res_dir):
    from charm_lensing.analysis_tools import source_distortion_ratio
    from scipy.stats import wasserstein_distance
    from charm_lensing.plotting import display_text
    from charm_lensing.analysis_tools import wmse, redchi2

    def cross_correlation(input, recon):
        return np.fft.ifft2(
            np.fft.fft2(input).conj() * np.fft.fft2(recon)
        ).real.max()

    def plot(s, x):
        from os.path import join
        from os import makedirs
        out_dir = join(res_dir, 'residuals')
        makedirs(out_dir, exist_ok=True)

        sky = jft.mean([sky_model(si) for si in s])

        dms = []
        for si in s:
            dm = np.zeros_like(plot_data)
            dm[mask] = data_model(si)
            dms.append(dm)
        mod_mean = jft.mean(dms)
        redchi_mean, redchi2_std = jft.mean_and_std(
            [redchi2(plot_data, m, std, plot_data.size) for m in dms])

        vals = dict(
            sdr=source_distortion_ratio(plot_sky, sky),
            wd=wasserstein_distance(plot_sky.reshape(-1), sky.reshape(-1)),
            cc=cross_correlation(plot_sky, sky),
        )

        fig, axes = plt.subplots(2, 3, figsize=(9, 6), dpi=300)
        ims = []
        axes[0, 0].set_title('Data')
        ims.append(axes[0, 0].imshow(plot_data, origin='lower'))
        axes[0, 1].set_title('Data model')
        ims.append(axes[0, 1].imshow(mod_mean, origin='lower'))
        axes[0, 2].set_title('Data residual')
        ims.append(axes[0, 2].imshow(
            (plot_data - mod_mean)/std, origin='lower', vmin=-3, vmax=3,
            cmap='RdBu_r'))
        chi = '\n'.join((
            f'MSE/var: {wmse(plot_data, mod_mean, std):.2f}',
            f'redChi2: {redchi_mean:.2f} +/- {redchi2_std:.2f}',
        ))
        display_text(axes[0, 2], chi)
        axes[1, 0].set_title('Sky')
        ims.append(axes[1, 0].imshow(plot_sky, origin='lower'))
        axes[1, 1].set_title('Sky model')
        ims.append(axes[1, 1].imshow(sky, origin='lower'))
        axes[1, 2].set_title('Sky residual')
        ims.append(axes[1, 2].imshow((plot_sky - sky)/plot_sky, origin='lower',
                                     vmin=-0.3, vmax=0.3, cmap='RdBu_r'))
        ss = '\n'.join([f'{k}: {v:.3f}' for k, v in vals.items()])
        display_text(axes[1, 2], ss)
        for ax, im in zip(axes.flatten(), ims):
            fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        fig.savefig(join(out_dir, f'{x.nit:02d}.png'), dpi=300)
        plt.close()

    return plot


plot = build_plot(
    d, comp_sky, mask, model, sky_model, res_dir)

pos_init = 0.1 * jft.Vector(jft.random_like(rec_key, like.domain))

cfg = ju.get_config('./JWST_config.yaml')
minimization_config = cfg['minimization']
kl_solver_kwargs = minimization_config.pop('kl_kwargs')
minimization_config['n_total_iterations'] = 15

samples, state = jft.optimize_kl(
    like,
    pos_init,
    key=key,
    kl_kwargs=kl_solver_kwargs,
    callback=plot,
    odir=res_dir,
    **minimization_config)
