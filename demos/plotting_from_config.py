import yaml
import argparse
import os

import pickle

import nifty8.re as jft

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, FuncNorm
from astropy import units as u

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
    build_plot_lens_system,
    build_plot_source,
    get_alpha_nonpar,
)
from jubik0.jwst.filter_projector import FilterProjector

from jubik0.jwst.color import Color, ColorRange

from charm_lensing.lens_system import build_lens_system

from sys import exit

from jax import config, devices
config.update('jax_default_device', devices('cpu')[0])


PLOT_RESIDUALS = False
PLOT_ALL_RECONSTRUCTION = False

parser = argparse.ArgumentParser()
parser.add_argument(
    "config",
    help="Config File",
    type=str,
    nargs='?',
    const=1,
    default='./demos/jwst_lens_config.yaml')
args = parser.parse_args()
config_path = args.config

cfg = yaml.load(open(config_path, 'r'), Loader=yaml.SafeLoader)
RES_DIR = cfg['files']['res_dir']


reconstruction_grid = build_reconstruction_grid_from_config(cfg)
# insert_ubik_energy_in_lensing(cfg, zsource=4.2)
insert_spaces_in_lensing(cfg)
lens_system = build_lens_system(cfg['lensing'])
sky_model = lens_system.get_forward_model_parametric()
if cfg['lens_only']:
    sky_model = lens_system.lens_plane_model.light_model

energy_cfg = cfg['grid']['energy_bin']
e_unit = getattr(u, energy_cfg.get('unit', 'eV'))
filter_keys = dict(
    e00='F2100W',
    e01='F1800W',
    e02='F1500W',
    e03='F1280W',
    e04='F1000W',
    e05='F770W',
    e06='F560W',
    e07='F444W',
    e08='F356W',
    e09='F277W',
    e10='F200W',
    e11='F150W',
    e12='F115W',
)
keys_and_colors = {
    filter_keys[f'e{ii:02d}']: ColorRange(
        Color(emin*e_unit), Color(emax*e_unit))
    for ii, (emin, emax) in enumerate(zip(energy_cfg.get('e_min'), energy_cfg.get('e_max')))}

filter_projector = FilterProjector(
    sky_domain=sky_model.target,
    keys_and_colors=keys_and_colors,
)
sky_model_with_keys = jft.Model(
    lambda x: filter_projector(sky_model(x)),
    init=sky_model.init
)


if PLOT_RESIDUALS:
    data_dict = {}
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
                data_transmission=jwst_data.transmission,
            )

    plot_residual = build_plot_sky_residuals(
        results_directory=RES_DIR,
        data_dict=data_dict,
        sky_model_with_key=sky_model_with_keys,
        small_sky_model=lens_system.lens_plane_model.light_model if cfg[
            'lens_only'] else lens_system.get_forward_model_parametric(),
        plotting_config=dict(
            norm=LogNorm,
            data_config=dict(norm=LogNorm),
            display_pointing=False,
            xmax_residuals=4,
        ),
    )


parametric_flag = lens_system.lens_plane_model.convergence_model.nonparametric() is not None
ll_alpha, ll_nonpar, sl_alpha, sl_nonpar = get_alpha_nonpar(lens_system)

plot_lens = build_plot_lens_system(
    RES_DIR,
    plotting_config=dict(
        norm_source=LogNorm,
        norm_lens=LogNorm,
        norm_source_nonparametric=LogNorm,
        # norm_mass=LogNorm,
    ),
    lens_system=lens_system,
    filter_projector=filter_projector,
    lens_light_alpha_nonparametric=(ll_alpha, ll_nonpar),
    source_light_alpha_nonparametric=(sl_alpha, sl_nonpar),
)

source_light = lens_system.source_plane_model.light_model
plot_source = build_plot_source(
    RES_DIR,
    plotting_config=dict(
        norm_source=LogNorm,
        norm_source_parametric=LogNorm,
        norm_source_nonparametric=LogNorm,
        extent=lens_system.source_plane_model.space.extend().extent,
    ),
    filter_projector=filter_projector,
    source_light_model=lens_system.source_plane_model.light_model,
    source_light_alpha=sl_alpha,
    source_light_parametric=lens_system.source_plane_model.light_model.parametric(),
    source_light_nonparametric=sl_nonpar,
    attach_name=''
)

cfm = lens_system.source_plane_model.light_model.nonparametric()
plot_deviations = build_plot_source(
    RES_DIR,
    plotting_config=dict(
        # norm_source=LogNorm,
        norm_source_parametric=LogNorm,
        norm_source_nonparametric=LogNorm,
        extent=lens_system.source_plane_model.space.extend().extent,
    ),
    filter_projector=filter_projector,
    source_light_model=cfm.spectral_deviations_distribution,
    # source_light_model=cfm.spectral_distribution,
    source_light_alpha=sl_alpha,
    source_light_parametric=lens_system.source_plane_model.light_model.parametric(),
    source_light_nonparametric=sl_nonpar,
    attach_name='dev_'
)


if PLOT_ALL_RECONSTRUCTION:
    n_iters = cfg['minimization']['n_total_iterations']
    for ii in range(n_iters):
        # if ii < 35:
        #     continue

        samp_stat = os.path.join(RES_DIR, f'position_{ii: 02d}')
        if not os.path.isfile(samp_stat):
            samp_stat = os.path.join(RES_DIR, f'samples_state_{ii:02d}.pkl')

        if os.path.isfile(samp_stat):
            with open(samp_stat, "rb") as f:
                samples, opt_vi_st = pickle.load(f)

            lens_path = os.path.join(RES_DIR, 'lens', f'{ii:02d}.png')
            res_path = os.path.join(RES_DIR, 'residuals', f'{ii:02d}.png')
            source_path = os.path.join(RES_DIR, 'source', f'{ii:02d}.png')
            dev_path = os.path.join(RES_DIR, 'source', f'dev_{ii:02d}.png')

            # if not os.path.isfile(lens_path):
            #     print('Plotting', ii)
            #     plot_lens(samples, opt_vi_st, parametric=parametric_flag)

            # if not os.path.isfile(res_path) and PLOT_RESIDUALS:
            #     print('Plotting residuals', ii)
            #     plot_residual(samples, opt_vi_st)

            if not os.path.isfile(source_path):
                print('Plotting source', ii)
                plot_source(samples, opt_vi_st)

            if not os.path.isfile(dev_path) and ii > 45:
                print('Plotting source deviations', ii)
                plot_deviations(samples, opt_vi_st)

else:
    samp_stat = os.path.join(RES_DIR, 'last.pkl')
    with open(samp_stat, "rb") as f:
        samples, opt_vi_st = pickle.load(f)
    plot_source(samples, opt_vi_st)
    plot_deviations(samples, opt_vi_st)
