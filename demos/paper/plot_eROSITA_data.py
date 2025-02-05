import matplotlib.pyplot as plt
import numpy as np
import pickle
from jax import linear_transpose, vmap
from matplotlib import patches
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
        ea = ju.instruments.erosita.erosita_response.calculate_erosita_effective_area(path_to_caldb, tm_ids, e_min, e_max)
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

    sat_min = {'log': [1.2e-9, 1.0e-10, 2.0e-10],
               "lin": [1e-10, 1e-10, 1e-10]}
    sat_max = {'log': [2.1e-7, 1.5e-7, 1.5e-7],
               "lin": [2.3e-8, 1.5e-8, 1.e-8]}
    #### LOG Plot
    plot_rgb(exposure_corrected_data,
             sat_min=sat_min['log'],
             sat_max=sat_max['log'],
             log=True,
             title='eROSITA LMC data', fs=18, pixel_measure=112,
             output_file=join(output_dir, 'log_expcor_eRSOITA_data_rgb.png'),
             alpha=0.0,
             pixel_factor=4,
             bbox_info=bbox_info,
             )

    #### Lin Plot
    plot_rgb(exposure_corrected_data,
             sat_min=sat_min['lin'],
             sat_max=sat_min['lin'],
             # log=True,
             title='eROSITA LMC data', fs=18, pixel_measure=112,
             output_file=join(output_dir, 'lin_expcor_eRSOITA_data_rgb.png'),
             alpha=0.0,
             pixel_factor=4,
             bbox_info=bbox_info,
             )
    plotting_kwargs_rec = {}
    for i in range(data.shape[0]):
        exposure_corrected_data_tm = data[i]/exposures[i]
        exposure_corrected_data_tm = exposure_corrected_data_tm / pixel_area
        mask_exp = exposures[i] == 0
        mask_data = np.isnan(exposure_corrected_data_tm)

        exposure_corrected_data_tm[mask_data] = 0
        exposure_corrected_data_tm[mask_exp] = 0
        plot(exposure_corrected_data_tm,
             pixel_factor=4,
             pixel_measure=112,
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
                              f'data_tm{i}.png'),
             **plotting_kwargs_rec
             )
        if i==0:
            plot_rgb(exposure_corrected_data_tm[:, 570: 770,  130: 330],
                     sat_min=sat_min['log'],
                     sat_max=sat_max['log'],
                     log=True,
                     title='eROSITA LMC data TM1', fs=32,
                     output_file=join(output_dir,
                                      'zoom_expcor_eRSOITA_data_rgb_tm1.png'),
                     alpha=0.0,
                     pixel_factor=4,
                     bbox_info=bbox_info,
                     )
    ### Zoom
    zoomed_expcor_data = exposure_corrected_data[:, 570: 770,  130: 330]
    plot_rgb(zoomed_expcor_data,
             sat_min=sat_min['log'],
             sat_max=sat_max['log'],
             log=True,
             title='eROSITA LMC data', fs=32,
             output_file=join(output_dir, 'zoom_expcor_eRSOITA_data_rgb.png'),
             alpha=0.0,
             pixel_factor=4,
             bbox_info=bbox_info,
             )
    x = np.log(exposure_corrected_data)
    x = ju._clip(x, np.log(sat_min['log']), np.log(sat_max['log']))
    x = np.moveaxis(x, 0, -1)
    plot_data = ju._norm_rgb_plot(x, minmax=(np.log(sat_min['log']),
                                             np.log(sat_max['log'])))
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.imshow(plot_data, origin="lower")


    ax.text(0.05, 0.95, 'eROSITA LMC data', fontsize=32,
             #fontfamily='cm',
             color='white',
             verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,
             bbox=(dict(facecolor='black', alpha=0.5, edgecolor='none')))
    rect = patches.Rectangle((130,570), 200, 200, facecolor='none', edgecolor='white')
    ax.add_patch(rect)
    plt.tight_layout()
    fig.savefig(join(output_dir, 'marked_expcor_eROSITA_data_rgb.png'),
                bbox_inches='tight', pad_inches=0)
    plt.close()

    # 30 Doradus C marked
    x = np.log(exposure_corrected_data)
    x = ju._clip(x, np.log(sat_min['log']), np.log(sat_max['log']))
    x = np.moveaxis(x, 0, -1)
    plot_data = ju._norm_rgb_plot(x, minmax=(np.log(sat_min['log']),
                                             np.log(sat_max['log'])))
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.imshow(plot_data, origin="lower")


    ax.text(0.05, 0.95, 'eROSITA LMC data', fontsize=18,
             #fontfamily='cm',
             color='white',
             verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,
             bbox=(dict(facecolor='black', alpha=0.5, edgecolor='none')))
    rect = patches.Rectangle((370,565), 120, 120, facecolor='none', edgecolor='white')
    ax.add_patch(rect)
    plt.tight_layout()
    fig.savefig(join(output_dir, '30D_marked_expcor_eROSITA_data_rgb.png'),
                bbox_inches='tight', pad_inches=0)
    plt.close()