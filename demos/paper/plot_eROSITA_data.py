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
    output_dir = "paper/"
    path_to_caldb = '../data/'
    config_path = join(output_dir, 'erosita_data_plotting_config.yaml')
    config_dict = ju.get_config(config_path)
    grid_info = config_dict["grid"]
    file_info = config_dict["files"]
    data_path = join(file_info['obs_path'], 'processed')
    tel_info = config_dict["telescope"]
    tm_ids = tel_info["tm_ids"]
    e_min = grid_info['energy_bin']['e_min']
    e_max = grid_info['energy_bin']['e_max']

    epix = grid_info['edim']
    spix = grid_info['sdim']
    pixel_area = (config_dict['telescope']['fov'] / config_dict['grid']['sdim']) **2 # density to flux

    data = []
    exposures = []
    for it, tm_id in enumerate(tm_ids):
        data_filenames = f'tm{tm_id}_' + file_info['output']
        exposure_filenames = f'tm{tm_id}_' + file_info['exposure']
        data_filenames = [join(data_path, f"{data_filenames.split('.')[0]}_emin{e}_emax{E}.fits")
                          for e, E in zip(e_min, e_max)]
        exposure_filenames = [join(data_path,
                                   f"{exposure_filenames.split('.')[0]}_emin{e}_emax{E}.fits")
                              for e, E in zip(e_min, e_max)]
        data.append([])
        exposures.append([])
        for e, output_filename in enumerate(data_filenames):
            with fits.open(output_filename) as hdul:
                data[it].append(hdul[0].data)
            with fits.open(exposure_filenames[e]) as hdul:
                exposures[it].append(hdul[0].data)

    data = np.array(data, dtype=int)
    exposures = np.array(exposures, dtype=float)
    exposures[exposures<=500] = 0 # FIXME FROM CONFIG Instroduce Exposure cut
    correct_exposures_for_effective_area = True
    if correct_exposures_for_effective_area:
        # from src.library.response import calculate_erosita_effective_area
        ea = ju.calculate_erosita_effective_area(path_to_caldb, tm_ids, e_min, e_max)
        exposures *= ea[:, :, np.newaxis, np.newaxis]

    summed_data = np.sum(data, axis=0)
    summed_exposure = np.sum(exposures, axis=0)
    exposure_corrected_data = summed_data/summed_exposure
    exposure_corrected_data = exposure_corrected_data / pixel_area
    mask_exp = summed_exposure == 0
    mask_data = np.isnan(exposure_corrected_data)

    exposure_corrected_data[mask_data] = 0
    exposure_corrected_data[mask_exp] = 0
    bbox_info = [(28, 16), 28, 160, 'black']

    #### LOG Plot
    plot_rgb(exposure_corrected_data,
             sat_min=[2e-12, 2e-12, 2e-12],
             sat_max=[1, 1, 1],
             log=True,
             title='eROSITA LMC data', fs=18, pixel_measure=112,
             output_file=join(output_dir, 'log_expcor_eRSOITA_data_rgb.png'),
             alpha=0.0,
             pixel_factor=4,
             bbox_info=bbox_info,
             )

    #### Lin Plot
    plot_rgb(exposure_corrected_data,
             sat_min=[1e-10, 1e-10, 1e-10],
             sat_max=[5e-8, 5e-8, 5e-8],
             # log=True,
             title='eROSITA LMC data', fs=18, pixel_measure=112,
             output_file=join(output_dir, 'lin_expcor_eRSOITA_data_rgb.png'),
             alpha=0.0,
             pixel_factor=4,
             bbox_info=bbox_info,
             )
