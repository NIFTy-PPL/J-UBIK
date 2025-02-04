import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cbook
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, \
    zoomed_inset_axes
import numpy as np
import pickle
from jax import linear_transpose, vmap
import jax.numpy as jnp
from jax import config
import jax
from pathlib import Path

import jubik0 as ju
from os.path import join, basename
import astropy.io.fits as fits
import nifty8.re as jft
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_eROSITA_image import plot, plot_rgb

config.update('jax_enable_x64', True)

# Script for plotting the data, position and reconstruction images
if __name__ == "__main__":
    results_path = "results/LMC-22092024-001V"
    config_name = "eROSITA_demo_ve_259.yaml"
    output_dir = ju.create_output_directory(join(results_path, 'paper'))
    config_path = join(results_path, config_name)
    config_dict = ju.get_config(config_path)
    tel_info = config_dict["telescope"]
    tm_ids = tel_info["tm_ids"]

    grid_info = config_dict["grid"]
    epix = grid_info['edim']
    spix = grid_info['sdim']
    e_min = grid_info['energy_bin']['e_min']
    e_max = grid_info['energy_bin']['e_max']

    with open(os.path.join(results_path, 'last.pkl'), "rb") as file:
        samples, _ = pickle.load(file)

    sky_model = ju.SkyModel(config_dict)
    sky = sky_model.create_sky_model()
    sky_dict = sky_model.sky_model_to_dict()

    file_info = config_dict['files']
    exposure_filenames = []
    for tm_id in tel_info['tm_ids']:
        exposure_filename = f'tm{tm_id}_' + file_info['exposure']
        [exposure_filenames.append(join(file_info['obs_path'],
                                        "processed",
                                        f"{Path(exposure_filename).stem}_emin{e}_emax{E}.fits"))
         for e, E in zip(e_min, e_max)]

    exposures = []
    tm_id = basename(exposure_filenames[0])[2]
    tm_exposures = []
    for file in exposure_filenames:
        if basename(file)[2] != tm_id:
            exposures.append(tm_exposures)
            tm_exposures = []
            tm_id = basename(file)[2]
        if basename(file)[2] == tm_id:
            if file.endswith('.npy'):
                tm_exposures.append(np.load(file))
            elif file.endswith('.fits'):
                tm_exposures.append(fits.open(file)[0].data)
            elif not (file.endswith('.npy') or file.endswith('.fits')):
                raise ValueError('exposure files should be in a .npy or .fits format!')
            else:
                raise FileNotFoundError(f'cannot find {file}!')
    exposures.append(tm_exposures)
    exposures = np.array(exposures, dtype="float64")
    summed_exposure = np.sum(exposures, axis=0)

    def mask(x):
        masked_x = x.at[summed_exposure<=500].set(0)
        return masked_x

    sat_min = {"log": [2e-12, 2e-12, 2e-12],
               "lin": [1e-10, 1e-10, 1e-10]}
    sat_max = {"log": [1, 1, 1],
               "lin": [2.3e-8, 1.5e-8, 1.e-8]}
    for key, op in sky_dict.items():
        op = jax.vmap(op)
        real_samples = op(samples.samples)
        real_mean = jnp.mean(real_samples, axis=0)
        if key != 'masked_diffuse':
            real_mean = mask(real_mean)
            pixel_measure = 112
            pixel_factor = 4
            bbox_info = [(28, 16), 28, 160, 'black']
            name = key
        else:
            pixel_measure = 20
            pixel_factor = 0.75
            bbox_info = [(3, 2), 3, 16, 'black']
            name = 'extended source'
        plot_rgb(real_mean,
                 sat_min=sat_min["log"],
                 sat_max=sat_max["log"],
                 pixel_factor=pixel_factor,
                 log=True,
                 title= f'reconstructed {name}',
                 fs=18,
                 pixel_measure=pixel_measure,
                 output_file=join(output_dir, f'rec_{key}_rgb.png'),
                 alpha=0.5,
                 bbox_info=bbox_info,
                 )
                 #LINEAR
                 #
        plot_rgb(real_mean,
                 sat_min=sat_min["lin"],
                 sat_max=sat_max["lin"],
                 pixel_factor=pixel_factor,
                 title= f'reconstructed {name}',
                 fs=18,
                 pixel_measure=pixel_measure,
                 output_file=join(output_dir, f'rec_{key}_rgb_lin.png'),
                 alpha=0.5,
                 bbox_info=bbox_info,
                 )

        if key == "sky":
            k = 0
            for sample in real_samples:
                plot_rgb(mask(sample),
                        sat_min=sat_min["lin"],
                        sat_max=sat_max["lin"],
                        pixel_factor=pixel_factor,
                        title= f'reconstructed {key}',
                        fs=18,
                        pixel_measure=pixel_measure,
                        output_file=join(output_dir, f'rec_{key}_rgb_lin_sample_{k}.png'),
                        alpha=0.5,
                        bbox_info=bbox_info,
                        )
                k = k+1
        plotting_kwargs_rec = {}
        plot(real_mean,
             pixel_factor=4,
             pixel_measure=112,
             fs=8,
                        title=['0.2-1.0 keV',
                               '1.0-2.0 keV',
                               '2.0-4.5 keV'],
                        logscale=True,
                        colorbar=True,
                        common_colorbar=True,
                        n_rows=1,
                        bbox_info=bbox_info,
                        output_file=join(output_dir,
                        f'rec_{key}.png'),
                        **plotting_kwargs_rec)
        real_std = jnp.std(real_samples, axis=0)
        if key != 'masked_diffuse':
            real_std = mask(real_std)
        plotting_kwargs_unc = {'cmap': 'jet', 'vmin': 1e-10, 'vmax':1e-6}
        plot(real_std,
             pixel_factor=4,
             pixel_measure=112,
             fs=8,
                        title=['0.2-1.0 keV',
                               '1.0-2.0 keV',
                               '2.0-4.5 keV'],
                        logscale=True,
                        colorbar=True,
                        common_colorbar=True,
                        n_rows=1,
                        bbox_info=bbox_info,
                        output_file=join(output_dir,
                        f'unc_{key}.png'),
                        **plotting_kwargs_unc)

        plot(real_std/real_mean,
             pixel_factor=4,
             pixel_measure=112,
             fs=12,
                        title=['0.2-1.0 keV',
                               '1.0-2.0 keV',
                               '2.0-4.5 keV'],
                        logscale=False,
                        colorbar=True,
                        #common_colorbar=True,
                        n_rows=1,
                        bbox_info=bbox_info,
                        output_file=join(output_dir,
                        f'rel_unc_{key}.png'),
                        **plotting_kwargs_unc)
    points_op = jax.vmap(sky_dict['points'])
    point_samples = points_op(samples.samples)
    real_points_cut = np.mean(point_samples, axis=0).at[\
        np.mean(point_samples, axis=0)<2.5e-9].set(0)
    diffuse_op = jax.vmap(sky_dict['diffuse'])
    diffuse_samples = diffuse_op(samples.samples)
    real_diffuse = np.mean(diffuse_samples, axis=0)
    masked_diffuse_op = jax.vmap(sky_dict['masked_diffuse'])
    masked_diffuse_samples = masked_diffuse_op(samples.samples)
    real_masked_diffuse = np.mean(masked_diffuse_samples, axis=0)

    cut_sky = real_points_cut + real_diffuse
    cut_sky = ju.add_masked_array(cut_sky, real_masked_diffuse,
                                  config_dict['priors']['masked_diffuse'])
    cut_sky = mask(cut_sky)

    pixel_measure = 112
    pixel_factor = 4
    bbox_info = [(28, 16), 28, 160, 'black']

    plot_rgb(cut_sky,
             sat_min=sat_min['lin'],
             sat_max=sat_max['lin'],
             sigma=None,
             title=f'reconstructed sky', fs=18, pixel_measure=pixel_measure,
             pixel_factor=pixel_factor,
             output_file=join(output_dir, f'cut_rec_sky_lin_rgb.png'),
             alpha=0.5,
             bbox_info=bbox_info
             )
    plot_rgb(cut_sky,
             sat_min=sat_min['log'],
             sat_max=sat_max['log'],
             pixel_factor=pixel_factor,
             log=True,
             # title= f'reconstructed {key}',
             fs=18,
             title=f'reconstructed sky',
             pixel_measure=pixel_measure,
             output_file=join(output_dir, f'cut_rec_sky_rgb.png'),
             alpha=0.5,
             bbox_info=bbox_info,
             )
    plot_rgb(mask(real_points_cut),
             sat_min=[2e-9, 2e-9, 2e-9],
             sat_max=[1.5e-8, 1.5e-8, 1.5e-8],
             sigma=0.5,
             title=f'reconstructed points', fs=18, pixel_measure=pixel_measure,
             output_file=join(output_dir, f'cut_rec_points_lin_rgb.png'),
             alpha=0.5,
             bbox_info=bbox_info,
             pixel_factor=pixel_factor,
             )
    plot_rgb(mask(real_points_cut),
             sat_min=sat_min['log'],
             sat_max=sat_max['log'],
             pixel_factor=pixel_factor,
             log=True,
             # title= f'reconstructed {key}',
             fs=18,
             title=f'reconstructed points',
             output_file=join(output_dir, f'cut_rec_points_rgb.png'),
             alpha=0.5,
             bbox_info=bbox_info,
             )

    plot_rgb(real_diffuse[:, 570: 770,  150: 350],
             sat_min=sat_min['lin'],
             sat_max=sat_max['lin'],
             sigma=None,
             title=f'TM1', fs=32,
             output_file=join(output_dir, f'zoom_cut_rec_sky_lin_rgb.png'),
             alpha=0.5,
             bbox_info=bbox_info
             )
    plot_rgb(real_diffuse[:, 570: 770,  150: 350],
             sat_min=sat_min['log'],
             sat_max=sat_max['log'],
             pixel_factor=pixel_factor,
             log=True,
             # title= f'reconstructed {key}',
             fs=18,
             title=f'TM1',
             pixel_measure=pixel_measure,
             output_file=join(output_dir, f'zoom_cut_rec_sky_rgb.png'),
             alpha=0.5,
             bbox_info=bbox_info,
             )



