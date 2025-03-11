import matplotlib.pyplot as plt
import numpy as np
import pickle
from jax import linear_transpose, vmap
import jax.numpy as jnp

import jubik0 as ju
from os.path import join
import astropy.io.fits as fits
import nifty8.re as jft
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_eROSITA_image import plot, plot_rgb


# Script for plotting the data, position and reconstruction images
if __name__ == "__main__":
    results_path = "results/Mock-17092024-001M"
    config_name = "eROSITA_demo.yaml"
    path_to_caldb = '../data/'
    output_dir = ju.create_output_directory(join(results_path, 'paper'))
    config_path = join(results_path, config_name)
    config_dict = ju.get_config(config_path)
    grid_info = config_dict["grid"]
    file_info = config_dict["files"]
    tel_info = config_dict["telescope"]
    tm_ids = tel_info["tm_ids"]
    e_min = grid_info['energy_bin']['e_min']
    e_max = grid_info['energy_bin']['e_max']
    data_path = join(results_path, 'data.pkl')
    processed_data_path = join(file_info['obs_path'], 'processed')

    epix = grid_info['edim']
    spix = grid_info['sdim']
    pixel_area = (config_dict['telescope']['fov'] / config_dict['grid']['sdim']) **2 # density to flux

    exposures = []
    for it, tm_id in enumerate(tm_ids):
        exposure_filenames = f'tm{tm_id}_' + file_info['exposure']
        exposure_filenames = [join(processed_data_path,
                                   f"{exposure_filenames.split('.')[0]}_emin{e}_emax{E}.fits")
                              for e, E in zip(e_min, e_max)]
        exposures.append([])
        for e, output_filename in enumerate(exposure_filenames):
            with fits.open(exposure_filenames[e]) as hdul:
                exposures[it].append(hdul[0].data)
    exposures = np.array(exposures, dtype=float)
    exposures[exposures<=tel_info['exp_cut']] = 0 # FIXME FROM CONFIG Instroduce Exposure cut
    correct_exposures_for_effective_area = True
    if correct_exposures_for_effective_area:
        # from src.library.response import calculate_erosita_effective_area
        ea = ju.calculate_erosita_effective_area(path_to_caldb, tm_ids, e_min, e_max)
        exposures *= ea[:, :, np.newaxis, np.newaxis]

    data = ju.load_masked_data_from_config(config_path)
    plottable_vector = jft.Vector({key: val.astype(float) for key, val
                                   in data.tree.items()})
    n_modules = len(data.tree)
    epix = grid_info['edim']
    spix = grid_info['sdim']
    response_dict = ju.build_erosita_response_from_config(config_path)
    mask = response_dict['mask']
    mask_adj = linear_transpose(mask,
                                np.zeros((n_modules, epix, spix, spix)))
    mask_adj_func = lambda x: mask_adj(x)[0]
    tms = plottable_vector.tree.keys()
    # Plotting the data
    unmasked_data = mask_adj_func(plottable_vector)
    plotting_kwargs = {'vmin':1e0, 'vmax':4e2}
    bbox_info = [(14, 8), 14,  40, 'black']
    for i in range(unmasked_data.shape[0]):
        plot(unmasked_data[i],
             pixel_factor=2,
             pixel_measure=28,
             fs=12,
                        title=['0.2-1.0 keV',
                               '1.0-2.0 keV',
                               '2.0-4.5 keV'],
                        logscale=True,
                        colorbar=True,
                        common_colorbar=True,
                        n_rows=1,
                        bbox_info=bbox_info,
                        output_file=join(output_dir,
                        f'tm{list(tms)[i]}_data.png'),
                        **plotting_kwargs)
    summed_data = np.sum(unmasked_data, axis=0)
    plot(summed_data,
         pixel_factor=2,
         pixel_measure=28,
         fs=12,
         title=['0.2-1.0 keV',
                '1.0-2.0 keV',
                '2.0-4.5 keV'],
         logscale=True,
         colorbar=True,
         common_colorbar=True,
         n_rows=1,
         output_file=join(output_dir,
                          f'summed_data.png'),
         bbox_info=bbox_info,
         **plotting_kwargs)
    plot_rgb(summed_data, sat_min=[0, 0, 0],
             sat_max=[4e2, 1e2, 1e1],
             sigma=None, log=True,
             title='simulated data', fs=18, pixel_measure=28,
             pixel_factor=2,
             output_file=join(output_dir, 'simulated_data_rgb.png'),
             alpha=0.5,
             bbox_info=bbox_info
             )

    summed_exposure = np.sum(exposures, axis=0)
    exposure_corrected_data = summed_data/summed_exposure
    exposure_corrected_data = exposure_corrected_data / pixel_area
    mask_exp = summed_exposure == 0
    mask_data = np.isnan(exposure_corrected_data)

    exposure_corrected_data = exposure_corrected_data.at[mask_data].set(0)
    exposure_corrected_data = exposure_corrected_data.at[mask_exp].set(0)

    sat_min = {'log': [1.2e-9, 1.0e-10, 2.0e-10],
               "lin": [1e-10, 1e-10, 1e-10]}
    sat_max = {'log': [2.1e-7, 1.5e-7, 1.5e-7],
               "lin": [2.3e-8, 1.5e-8, 1.e-8]}
    #### LOG Plot

    plot_rgb(exposure_corrected_data,
             sat_min=sat_min['log'],
             sat_max=sat_max['log'],
             log=True,
             title='simulated data', fs=20, pixel_measure=28,
             pixel_factor=2,
             output_file=join(output_dir, 'log_expcor_mock_data_rgb.png'),
             alpha=0.5,
             bbox_info=bbox_info,
             )

    #### Lin Plot
    plot_rgb(exposure_corrected_data,
             sat_min=sat_min['lin'],
             sat_max=sat_min['lin'],
             # log=True,
             title='simulated data', fs=20, pixel_measure=28,
             pixel_factor=2,
             output_file=join(output_dir, 'lin_expcor_mock_data_rgb.png'),
             alpha=0.5,
             bbox_info=bbox_info,
             )
    plotting_kwargs = {'vmin':2e-9, 'vmax':8e-7}
    for i in range(unmasked_data.shape[0]):
        exposure_corrected_data = unmasked_data[i]/exposures[i]
        exposure_corrected_data = exposure_corrected_data / pixel_area#
        mask_exp = exposures[i] == 0
        mask_data = np.isnan(exposure_corrected_data)

        exposure_corrected_data = exposure_corrected_data.at[mask_data].set(0)
        exposure_corrected_data = exposure_corrected_data.at[mask_exp].set(0)
        plot(exposure_corrected_data,
             pixel_measure=28,
             fs=12,
             pixel_factor=2,
                        title=['0.2-1.0 keV',
                               '1.0-2.0 keV',
                               '2.0-4.5 keV'],
                        colorbar=True,
             common_colorbar=True,
             logscale=True,
                        n_rows=1,
                        bbox_info=bbox_info,
                        output_file=join(output_dir,
                        f'exp_cor_tm{list(tms)[i]}_data.png'),
                        **plotting_kwargs)

