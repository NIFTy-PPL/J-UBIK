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
    tel_info = config_dict["telescope"]
    tm_ids = tel_info["tm_ids"]
    e_min = grid_info['energy_bin']['e_min']
    e_max = grid_info['energy_bin']['e_max']
    data_path = join(file_info['obs_path'], 'processed')

    epix = grid_info['edim']
    spix = grid_info['sdim']
    exposures = []
    for it, tm_id in enumerate(tm_ids):
        exposure_filenames = f'tm{tm_id}_' + file_info['exposure']
        exposure_filenames = [join(data_path,
                                   f"{exposure_filenames.split('.')[0]}_emin{e}_emax{E}.fits")
                              for e, E in zip(e_min, e_max)]
        exposures.append([])
        for e, output_filename in enumerate(exposure_filenames):
            with fits.open(exposure_filenames[e]) as hdul:
                exposures[it].append(hdul[0].data)

    exposures = np.array(exposures, dtype=float)
    exposures[exposures<=500] = 0 # FIXME FROM CONFIG Instroduce Exposure cut
    summed_exposure = np.sum(exposures, axis=0)
    correct_exposures_for_effective_area = True
    if correct_exposures_for_effective_area:
        # from src.library.response import calculate_erosita_effective_area
        ea = ju.instruments.erosita.erosita_response.calculate_erosita_effective_area(path_to_caldb, tm_ids, e_min, e_max)
        exposures *= ea[:, :, np.newaxis, np.newaxis]

    plotting_kwargs ={}
    # Plotting the data
    bbox_info = [(28, 16), 28, 160, 'black']
    for i in range(exposures.shape[0]):
        plot(exposures[i],
             pixel_measure=112,
             fs=12,
                        title=['0.2-1.0 keV',
                               '1.0-2.0 keV',
                               '2.0-4.5 keV'],
                        logscale=True,
                        colorbar=True,
                        common_colorbar=True,
                        n_rows=1,
                        alpha=0.5,
                        bbox_info= bbox_info,
                        pixel_factor=4,
                        cmap='plasma',
                        output_file=join(output_dir,
                        f'eROSITA_tm{tm_ids[i]}_exposure.png'),
                        **plotting_kwargs)
    bbox_info = [(28, 16), 28,  160, 'black']
    plot(summed_exposure,
         pixel_measure=112,
         fs=12,
         title=['0.2-1.0 keV',
                '1.0-2.0 keV',
                '2.0-4.5 keV'],
         logscale=True,
         colorbar=True,
         common_colorbar=True,
         n_rows=1,
         output_file=join(output_dir,
                          f'summed_eROSITA_exposure.png'),
         pixel_factor=4,
         bbox_info=bbox_info,
         alpha=0.5,
         vmax=1e6,
         vmin=1e4,
         dpi=300,
         cmap='plasma',
         **plotting_kwargs)
