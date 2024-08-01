import yaml
from functools import reduce

import nifty8.re as jft
from jax import random
import jax.numpy as jnp

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, FuncNorm
from astropy import units as u

import jubik0 as ju
from jubik0.library.likelihood import (
    connect_likelihood_to_model, build_gaussian_likelihood)
from jubik0.jwst.jwst_data import JwstData
from jubik0.jwst.masking import get_mask_from_index_centers
from jubik0.jwst.config_handler import (
    build_reconstruction_grid_from_config,
    build_coordinates_correction_prior_from_config,
    build_filter_zero_flux,
    insert_spaces_in_lensing)
from jubik0.jwst.wcs import (subsample_grid_centers_in_index_grid)
from jubik0.jwst.jwst_data_model import build_data_model
from jubik0.jwst.jwst_plotting import (
    build_plot_sky_residuals,
    build_color_components_plotting,
    build_plot_lens_system, get_alpha_nonpar,
    rgb_plotting,
)
from jubik0.jwst.filter_projector import FilterProjector

from jubik0.jwst.color import Color, ColorRange

from charm_lensing.lens_system import build_lens_system

import argparse
import os

from sys import exit

parser = argparse.ArgumentParser()
parser.add_argument(
    "config",
    help="Config File",
    type=str,
    nargs='?',
    const=1,
    default='./demos/jwst_lens_config.yaml')
parser.add_argument(
    '--cpu',
    action='store_true',
    help='Use CPU',
    default=False
)
args = parser.parse_args()
config_path = args.config

cfg = yaml.load(open(config_path, 'r'), Loader=yaml.SafeLoader)
RES_DIR = cfg['files']['res_dir']
os.makedirs(RES_DIR, exist_ok=True)
ju.save_local_packages_hashes_to_txt(
    ['nifty8', 'charm_lensing', 'jubik0'],
    os.path.join(RES_DIR, 'hashes.txt'))
ju.save_config_copy_easy(config_path, os.path.join(RES_DIR, 'config.yaml'))

if cfg['cpu']:
    from jax import config, devices
    config.update('jax_default_device', devices('cpu')[0])


reconstruction_grid = build_reconstruction_grid_from_config(cfg)
# insert_ubik_energy_in_lensing(cfg, zsource=4.2)
insert_spaces_in_lensing(cfg)
lens_system = build_lens_system(cfg['lensing'])
sky_model = lens_system.get_forward_model_parametric()
if cfg['lens_only']:
    sky_model = lens_system.lens_plane_model.light_model

energy_cfg = cfg['grid']['energy_bin']
e_unit = getattr(u, energy_cfg.get('unit', 'eV'))
keys_and_colors = {
    f'e{ii:02d}': ColorRange(Color(emin*e_unit), Color(emax*e_unit))
    for ii, (emin, emax) in enumerate(zip(energy_cfg.get('e_min'), energy_cfg.get('e_max')))}

filter_projector = FilterProjector(
    sky_domain=sky_model.target,
    keys_and_colors=keys_and_colors,
)
sky_model_with_keys = jft.Model(
    lambda x: filter_projector(sky_model(x)),
    init=sky_model.init
)

data_dict = {}
likelihoods = []
for fltname, flt in cfg['files']['filter'].items():
    for ii, filepath in enumerate(flt):
        print(fltname, ii, filepath)
        jwst_data = JwstData(filepath)
        # print(fltname, jwst_data.half_power_wavelength)

        ekey = filter_projector.get_key(jwst_data.pivot_wavelength)
        data_key = f'{fltname}_{ekey}_{ii}'

        # Loading data, std, and mask.
        psf_ext = int(cfg['telescope']['psf']['psf_pixels'] // 2)
        mask = get_mask_from_index_centers(
            np.squeeze(subsample_grid_centers_in_index_grid(
                reconstruction_grid.world_extrema(ext=(psf_ext, psf_ext)),
                jwst_data.wcs,
                reconstruction_grid.wcs,
                1)),
            reconstruction_grid.shape)
        mask *= jwst_data.nan_inside_extrema(
            reconstruction_grid.world_extrema(ext=(psf_ext, psf_ext)))
        data = jwst_data.data_inside_extrema(
            reconstruction_grid.world_extrema(ext=(psf_ext, psf_ext)))
        std = jwst_data.std_inside_extrema(
            reconstruction_grid.world_extrema(ext=(psf_ext, psf_ext)))

        # plt.imshow(data)
        # plt.title(f'{fltname} {ii} {data[mask].sum()}')
        # plt.show()

        data_model = build_data_model(
            {ekey: sky_model_with_keys.target[ekey]},
            reconstruction_grid=reconstruction_grid,
            subsample=cfg['telescope']['rotation_and_shift']['subsample'],

            rotation_and_shift_kwargs=dict(
                data_dvol=jwst_data.dvol,
                data_wcs=jwst_data.wcs,
                data_model_type=cfg['telescope']['rotation_and_shift']['model'],
                shift_and_rotation_correction=dict(
                    domain_key=data_key + '_correction',
                    priors=build_coordinates_correction_prior_from_config(
                        cfg, jwst_data.filter, ii),
                )
            ),

            psf_kwargs=dict(
                camera=jwst_data.camera,
                filter=jwst_data.filter,
                center_pixel=jwst_data.wcs.index_from_wl(
                    reconstruction_grid.center)[0],
                webbpsf_path=cfg['telescope']['psf']['webbpsf_path'],
                psf_library_path=cfg['telescope']['psf']['psf_library'],
                fov_pixels=cfg['telescope']['psf']['psf_pixels'],
            ),

            transmission=jwst_data.transmission,

            data_mask=mask,

            world_extrema=reconstruction_grid.world_extrema(
                ext=(psf_ext, psf_ext)),

            zero_flux=dict(
                dkey=data_key,
                zero_flux=build_filter_zero_flux(cfg, jwst_data.filter),
            ),
        )

        data_dict[data_key] = dict(
            index=filter_projector.keys_and_index[ekey],
            data=data,
            std=std,
            mask=mask,
            data_model=data_model,
            data_dvol=jwst_data.dvol,
        )

        likelihood = build_gaussian_likelihood(
            jnp.array(data[mask], dtype=float),
            jnp.array(std[mask], dtype=float))
        likelihood = likelihood.amend(
            data_model, domain=jft.Vector(data_model.domain))
        likelihoods.append(likelihood)


model = sky_model_with_keys
likelihood = reduce(lambda x, y: x+y, likelihoods)
likelihood = connect_likelihood_to_model(likelihood, model)

# PLOTTING
parametric_flag = lens_system.lens_plane_model.convergence_model.nonparametric() is not None
ll_alpha, ll_nonpar, sl_alpha, sl_nonpar = get_alpha_nonpar(lens_system)


plot_lens = build_plot_lens_system(
    RES_DIR,
    plotting_config=dict(
        norm_source=LogNorm,
        norm_lens=LogNorm,
        # norm_mass=LogNorm,
    ),
    lens_system=lens_system,
    filter_projector=filter_projector,
    lens_light_alpha_nonparametric=(ll_alpha, ll_nonpar),
    source_light_alpha_nonparametric=(sl_alpha, sl_nonpar),
)
plot_residual = build_plot_sky_residuals(
    results_directory=RES_DIR,
    data_dict=data_dict,
    sky_model_with_key=sky_model_with_keys,
    small_sky_model=lens_system.lens_plane_model.light_model if cfg[
        'lens_only'] else lens_system.get_forward_model_parametric(),
    plotting_config=dict(
        norm=LogNorm, data_config=dict(norm=LogNorm))
)
plot_color = build_color_components_plotting(
    lens_system.source_plane_model.light_model.nonparametric(), RES_DIR, substring='source')


if cfg.get('prior_samples'):
    test_key, _ = random.split(random.PRNGKey(42), 2)

    def filter_data(datas: dict):
        filters = list()

        for kk, vv in datas.items():
            f = kk.split('_')[0]
            if f not in filters:
                filters.append(f)
                yield kk, vv

    prior_dict = {kk: vv for kk, vv in filter_data(data_dict)}
    plot_prior = build_plot_sky_residuals(
        results_directory=RES_DIR,
        data_dict=prior_dict,
        sky_model_with_key=sky_model_with_keys,
        small_sky_model=lens_system.lens_plane_model.light_model if cfg[
            'lens_only'] else lens_system.get_forward_model_parametric(),
        plotting_config=dict(
            norm=LogNorm,
            data_config=dict(norm=LogNorm),
            display_chi2=False,
            display_pointing=False,
            std_relative=False,
        )
    )

    nsamples = cfg.get('prior_samples') if cfg.get('prior_samples') else 3
    for ii in range(nsamples):
        test_key, _ = random.split(test_key, 2)
        position = likelihood.init(test_key)
        while isinstance(position, jft.Vector):
            position = position.tree

        # if not cfg['lens_only']:
        #     plot_lens(position, None, parametric=parametric_flag)
        plot_prior(position)
        # plot_color(position)


def plot(samples: jft.Samples, state: jft.OptimizeVIState):
    print(f'Plotting: {state.nit}')
    plot_residual(samples, state)
    plot_color(samples, state)
    if not cfg['lens_only']:
        plot_lens(samples, state, parametric=parametric_flag)


cfg_mini = ju.get_config(config_path)["minimization"]
n_dof = ju.calculate_n_constrained_dof(likelihood)
minpars = ju.MinimizationParser(cfg_mini, n_dof, verbose=False)
key = random.PRNGKey(cfg_mini.get('key', 42))
key, rec_key = random.split(key, 2)
pos_init = 0.1 * jft.Vector(jft.random_like(rec_key, likelihood.domain))


print(f'Results: {RES_DIR}')
samples, state = jft.optimize_kl(
    likelihood,
    pos_init,
    key=rec_key,
    callback=plot,
    odir=RES_DIR,
    n_total_iterations=cfg_mini['n_total_iterations'],
    n_samples=minpars.n_samples,
    sample_mode=minpars.sample_mode,
    draw_linear_kwargs=minpars.draw_linear_kwargs,
    nonlinearly_update_kwargs=minpars.nonlinearly_update_kwargs,
    kl_kwargs=minpars.kl_kwargs,
    resume=cfg_mini.get('resume', False),
)

exit()
if __name__ == "__main__":

    rgb_plotting(
        lens_system, samples, three_filter_names=('f1000w', 'f770w', 'f560w')
    )

    llm = lens_system.lens_plane_model.light_model
    for comp, fltname in zip(
            llm.nonparametric().components, cfg['files']['filter'].keys()):
        print(fltname)
        distro = comp.parametric().prior
        prior_keys = comp.parametric().model.prior_keys
        ms, ss = jft.mean_and_std([distro(s) for s in samples])
        for (k, _), m, s in zip(prior_keys, ms, ss):
            print(k, m, s, sep='\t')
        print()

    from jubik0.jwst.jwst_plotting import (_get_data_model_and_chi2,
                                           _get_sky_or_skies)
    sky = _get_sky_or_skies(samples, sky_model_with_keys)

    llm = lens_system.lens_plane_model.light_model
    sky_ll_with_keys = jft.Model(
        lambda x: filter_projector(llm(x)),
        init=llm.init
    )
    sky_ll = _get_sky_or_skies(samples, sky_ll_with_keys)

    slm = lens_system.get_forward_model_parametric(only_source=True)
    sky_sl_with_keys = jft.Model(
        lambda x: filter_projector(slm(x)),
        init=slm.init
    )
    sky_sl = _get_sky_or_skies(samples, sky_sl_with_keys)

    def get_pixel_radius_from_max_value(sky):
        shape = sky.shape
        xx, yy = np.meshgrid(*(np.arange(shape[1]), np.arange(shape[0])))
        xmax, ymax = np.unravel_index(np.argmax(sky), sky.shape)
        return np.hypot(yy-xmax, xx-ymax)

    radii = []
    radial_fm = []
    radial_data = []
    radial_ll = []
    radial_sl = []
    zz = {'f2100w': 221.56061654,
          'f1800w': 92.48311537,
          'f1500w': 42.64578287,
          'f1280w': 24.97585328,
          'f1000w': 13.98030351,
          'f770w': 4.2036279,
          'f560w': 1.22458536,
          'f444w': 0.14007522,
          'f356w': 0.01034513,
          'f277w': 0.01760561,
          'f200w': 0.02736996,
          'f150w': 0.00221736,
          'f115w': 0.11671709}

    for dkey, dval in data_dict.items():
        flt, ekey, _ = dkey.split('_')
        print(flt)
        index = int(ekey.split('e')[1])

        data = dval['data']
        data_model = dval['data_model']

        full_model_mean, _ = _get_data_model_and_chi2(
            samples,
            sky,
            data_model=data_model,
            data=data,
            mask=dval['mask'],
            std=dval['std'])
        ll_model_mean, _ = _get_data_model_and_chi2(
            samples,
            sky_ll,
            data_model=data_model,
            data=data,
            mask=dval['mask'],
            std=dval['std'])
        sl_model_mean, _ = _get_data_model_and_chi2(
            samples,
            sky_sl,
            data_model=data_model,
            data=data,
            mask=dval['mask'],
            std=dval['std'])

        rel_r = get_pixel_radius_from_max_value(ll_model_mean)
        max = int(np.ceil(np.max(rel_r)))
        pixel_radius = np.linspace(0, np.max(rel_r), max)
        ddist = np.sqrt(dval['data_dvol']).to(u.arcsec).value

        pr = []  # physical radius
        fm_radial = []
        ll_radial = []
        sl_radial = []
        data_radial = []
        for ii in range(pixel_radius.shape[0]-1):
            mask = ((pixel_radius[ii] < rel_r) *
                    (rel_r < pixel_radius[ii+1]) *
                    dval['mask'])
            pr.append(ii*ddist)
            fm_radial.append(np.nanmean(full_model_mean[mask]))
            ll_radial.append(np.nanmean(ll_model_mean[mask]))
            sl_radial.append(np.nanmean(sl_model_mean[mask]))
            data_radial.append(np.nanmean(data[mask]))
        radii.append(np.array(pr))
        radial_fm.append(np.array(fm_radial))
        radial_ll.append(np.array(ll_radial))
        radial_sl.append(np.array(sl_radial))
        radial_data.append(np.array(data_radial))

    fig, axes = plt.subplots(2, len(sky_model_with_keys.target))
    axes = axes.T
    jj = -1
    for dkey, pr,  data_radial, fm_radial, ll_radial, sl_radial in zip(
            data_dict.keys(), radii, radial_data, radial_fm, radial_ll, radial_sl):
        maxindex = fm_radial.shape[0]
        flt, ekey, _ = dkey.split('_')
        jj += 1
        ax = axes[jj]
        ax[0].set_title(flt)
        ax[0].plot(pr, data_radial-zz[flt], label='data', color='black')
        ax[0].plot(pr, fm_radial-zz[flt], label='full_model')
        ax[0].plot(pr, ll_radial-zz[flt], label='lens_light_model')
        ax[0].plot(pr, sl_radial-zz[flt], label='source_light_model')
        ax[0].set_ylim(bottom=0.1, top=100)
        ax[0].loglog()
        ax[0].legend()
        ax[1].plot(pr, data_radial-fm_radial, label='data - full_model')
        ax[1].plot(pr, data_radial-ll_radial, label='data - lens_light_model')
        ax[1].plot(pr, data_radial-sl_radial,
                   label='data - source_light_model')
        ax[1].set_ylim(bottom=-0.5, top=0.5)
        ax[1].legend()
    plt.show()
