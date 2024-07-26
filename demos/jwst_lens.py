import yaml
from functools import reduce

import nifty8.re as jft
from jax import random
import jax.numpy as jnp

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy import units as u

import jubik0 as ju
from jubik0.library.likelihood import (
    connect_likelihood_to_model, build_gaussian_likelihood)
from jubik0.jwst.jwst_data import JwstData
from jubik0.jwst.masking import get_mask_from_index_centers
from jubik0.jwst.config_handler import (
    build_reconstruction_grid_from_config,
    build_coordinates_correction_prior_from_config,
    insert_spaces_in_lensing)
from jubik0.jwst.wcs import (subsample_grid_centers_in_index_grid)
from jubik0.jwst.jwst_data_model import build_data_model
from jubik0.jwst.jwst_plotting import (
    build_plot_sky_residuals, build_color_components_plotting, build_plot_lens_system, get_alpha_nonpar)
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
                zero_flux=dict(prior=cfg['telescope']['zero_flux']['prior']),
            ),
        )

        data_dict[data_key] = dict(
            index=filter_projector.keys_and_index[ekey],
            data=data,
            std=std,
            mask=mask,
            data_model=data_model,
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
    small_sky_model=lens_system.get_forward_model_parametric(),
    plotting_config=dict(
        norm=LogNorm, data_config=dict(norm=LogNorm))
)
plot_color = build_color_components_plotting(
    lens_system.source_plane_model.light_model.nonparametric(), RES_DIR, substring='source')


if cfg.get('prior_samples') is not None:
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
        small_sky_model=lens_system.get_forward_model_parametric(),
        plotting_config=dict(
            norm=LogNorm,
            data_config=dict(norm=LogNorm),
            display_chi2=False,
            display_pointing=False,
            std_relative=False,
        )
    )

    nsamples = 3 if cfg.get(
        'prior_samples') is None else cfg.get('prior_samples')
    for ii in range(nsamples):
        test_key, _ = random.split(test_key, 2)
        position = likelihood.init(test_key)
        while isinstance(position, jft.Vector):
            position = position.tree

        plot_lens(position, None, parametric=parametric_flag)
        plot_prior(position)
        # plot_color(position)


def plot(samples: jft.Samples, state: jft.OptimizeVIState):
    print(f'Plotting: {state.nit}')
    plot_residual(samples, state)
    plot_lens(samples, state, parametric=parametric_flag)
    plot_color(samples, state)


cfg_mini = ju.get_config(config_path)["minimization"]
n_dof = ju.calculate_n_constrained_dof(likelihood)
minpars = ju.MinimizationParser(cfg_mini, n_dof)
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
    sl = lens_system.source_plane_model.light_model
    ms, mss = jft.mean_and_std([sl(si) for si in samples])

    print(filter_projector.keys_and_colors)
    print(filter_projector.keys_and_index)

    # for ii, mo in enumerate(lens_system.lens_plane_model.light_model.nonparametric()._comps):
    #     para = mo.parametric()[0]
    #     keys = [k[0] for k in para._model.prior_keys]
    #     pp, pps = jft.mean_and_std([para._prior(s) for s in samples])
    #     print(ii)
    #     print(
    #         '\n'.join((f'{k:10} {p}+-{ps}' for
    #                    k, p, ps in zip(keys, pp, pps)))
    #     )
    #     print()

    from astropy import units, cosmology

    sextent = np.array(lens_system.source_plane_model.space.extend().extent)
    extent_kpc = (np.tan(sextent * units.arcsec) *
                  cosmology.Planck13.angular_diameter_distance(4.2).to(
                      units.kpc)
                  ).value

    vmin, vmax = 1e-5, 2.3e-3

    rgb = np.zeros((384, 384, 3))
    # rgb[:, :, 0] = ms[0]
    rgb[:, :, 1] = ms[0]
    rgb[:, :, 2] = ms[1]

    rgb = rgb / np.max(rgb)
    rgb = np.sqrt(rgb)

    from charm_lensing.plotting import display_scalebar, display_text
    import matplotlib.font_manager as fm
    fig, ax = plt.subplots(1, 1, figsize=(11.5, 10))
    im = ax.imshow(rgb, origin='lower', extent=extent_kpc)
    display_scalebar(ax, dict(size=5, unit='kpc',
                     font_properties=fm.FontProperties(size=24)))
    display_text(ax,
                 text=dict(s='f444w', color='green',
                           fontproperties=fm.FontProperties(size=30)),
                 keyword='top_right',
                 y_offset_ticker=0,)
    display_text(ax,
                 text=dict(s='f356w', color='blue',
                           fontproperties=fm.FontProperties(size=30)),
                 keyword='top_right',
                 y_offset_ticker=1,
                 )
    ax.set_xlim(-5.5, 6)
    ax.set_ylim(-4, 6)
    plt.axis('off')  # Turn off axis
    plt.tight_layout()
    plt.savefig('f444w_f356w_source.png')
    plt.close()
