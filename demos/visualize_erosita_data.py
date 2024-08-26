import matplotlib.pyplot as plt
import numpy as np

import jubik0 as ju
from os.path import join
import astropy.io.fits as fits
from jubik0.library.response import calculate_erosita_effective_area

if __name__ == "__main__":
    config_path = "eROSITA_config.yaml"
    config_dict = ju.get_config(config_path)
    path_to_caldb = '../data/'
    file_info = config_dict["files"]
    grid_info = config_dict["grid"]
    tm_ids = config_dict["telescope"]["tm_ids"]
    e_min = grid_info['energy_bin']['e_min']
    e_max = grid_info['energy_bin']['e_max']
    data_path = join(file_info['obs_path'], 'processed')

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
        ea = calculate_erosita_effective_area(path_to_caldb, tm_ids, e_min, e_max)
        exposures *= ea[:, :, np.newaxis, np.newaxis]

    ju.plot_result(data[0], logscale=True, n_rows=1,n_cols=3, figsize=(15, 5),
                   adjust_figsize=False, common_colorbar=False)
    ju.plot_result(exposures[0], logscale=True, n_rows=1, n_cols=3, adjust_figsize=False,
                   figsize=(15, 5), common_colorbar=False)

    summed_data = np.sum(data, axis=0)
    summed_exposure = np.sum(exposures, axis=0)
    exposure_corrected_data = summed_data/summed_exposure
    exposure_corrected_data = exposure_corrected_data / pixel_area
    mask_data = np.isnan(exposure_corrected_data)
    mask_exp = summed_exposure == 0
    exposure_corrected_data[mask_data] = 0
    exposure_corrected_data[mask_exp] = 0
    ju.plot_result(exposure_corrected_data, logscale=True, n_rows=1, n_cols=3, adjust_figsize=False,
                   figsize=(15, 5), common_colorbar=False, output_file='expcordata.png')
    sat_min = [3e-10, 3e-10, 3e-10]

    # for setting the value
    maxim = [np.max(exposure_corrected_data[i]) for i in range(3)]
    for i in range(3):
        print(f"Max Bin {i}", maxim[i])

    # maxima set by eye
    sat_max = [1.5e-8, 0.9e-8, 1.0e-8]

    ju.plot_rgb(exposure_corrected_data, "blib", sat_min, sat_max, sigma=None, log=False)
    ju.plot_rgb(exposure_corrected_data, "blib_log", log=True)
    ju.plot_rgb(summed_data, "blib", sat_min, sat_max, sigma=None, log=False)
    ju.plot_rgb(exposure_corrected_data, "blib_log", log=True)
    ju.plot_result(exposures[0], logscale=True, n_cols=3, adjust_figsize=True, common_colorbar=True)

