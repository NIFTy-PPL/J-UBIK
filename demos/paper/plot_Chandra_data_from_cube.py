import matplotlib.pyplot as plt
from matplotlib import patches
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
    data_path = "./doradus/exp_correct_data.npy"
    exposure_path = "./doradus/exposures.npy"

    data = np.load(data_path)
    exposures = np.load(exposure_path)

    exposure_corrected_data = np.sum(np.swapaxes(data, -1, -2), axis=0)
    # exposures = np.array(exposures, dtype=float)
    # exposures[exposures<=500] = 0 # FIXME FROM CONFIG Instroduce Exposure cut
    # correct_exposures_for_effective_area = True
    # if correct_exposures_for_effective_area:
    # from src.library.response import calculate_erosita_effective_area
    #    ea = ju.calculate_erosita_effective_area(path_to_caldb, tm_ids, e_min, e_max)
    #    exposures *= ea[:, :, np.newaxis, np.newaxis]

    summed_exposure = np.sum(np.swapaxes(exposures, -1, -2), axis=0)
    # exposure_corrected_data = summed_data/summed_exposure
    exposure_corrected_data = exposure_corrected_data
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
             title='Chandra LMC data', fs=18, pixel_measure=112,
             output_file=join(output_dir, 'log_expcor_chandra_data_rgb.png'),
             alpha=0.0,
             pixel_factor=4,
             bbox_info=bbox_info,
             )

    #### Lin Plot
    plot_rgb(exposure_corrected_data,
             sat_min=[1e-10, 1e-10, 1e-10],
             #sat_max=[3.4e-7, 2.2e-7, 1.5e-7],
             sat_max=[2.3e-8, 1.5e-8, 1.e-8],
             # log=True,
             title='Chandra LMC data', fs=18, pixel_measure=112,
             output_file=join(output_dir, 'lin_expcor_chandra_data_rgb.png'),
             alpha=0.0,
             pixel_factor=4,
             bbox_info=bbox_info,
             )
    plotting_kwargs_rec = {}
    ### Zoom
    zoomed_expcor_data = exposure_corrected_data[:, 570: 770, 150: 350]
    plot_rgb(zoomed_expcor_data,
             sat_min=[1e-10, 1e-10, 1e-10],
             sat_max=[2.3e-8, 1.5e-8, 1.e-8],
             #sat_max=[3.4e-7, 2.2e-7, 1.5e-7],
             # log=True,
             title='Chandra LMC data', fs=32,
             output_file=join(output_dir, 'zoom_expcor_Chandra_data_rgb.png'),
             alpha=0.0,
             pixel_factor=4,
             bbox_info=bbox_info,
             )
