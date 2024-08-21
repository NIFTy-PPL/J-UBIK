import matplotlib.pyplot as plt
import numpy as np

import jubik0 as ju
from os.path import join
import astropy.io.fits as fits
from jubik0.library.response import calculate_erosita_effective_area

if __name__ == "__main__":
    config_path = "paper/erosita_data_plotting_config.yaml"
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

    # ju.plot_result(data[0], logscale=True, n_rows=1,n_cols=3, figsize=(15, 5),
    #                adjust_figsize=False, common_colorbar=False)
    # ju.plot_result(exposures[0], logscale=True, n_rows=1, n_cols=3, adjust_figsize=False,
    #                figsize=(15, 5), common_colorbar=False)

    summed_data = np.sum(data, axis=0)
    summed_exposure = np.sum(exposures, axis=0)

    non_zero_mask = summed_exposure != 0

    exposure_corrected_data = np.zeros_like(summed_data, dtype=float)

    exposure_corrected_data[non_zero_mask] = summed_data[non_zero_mask] / (summed_exposure[non_zero_mask]*pixel_area)

    ju.plot_result(exposure_corrected_data, logscale=True, n_rows=1, n_cols=3, adjust_figsize=False,
                   figsize=(15, 5), common_colorbar=False, output_file='../expcordata.png')

    exp_delta_data = exposure_corrected_data.copy()
    from functools import reduce
    from operator import mul
    non_zero_elements = summed_exposure[summed_exposure != 0]
    detection_threshold = 4.6/(np.max(non_zero_elements)* pixel_area)
    print(f"Detection Threshold is {detection_threshold}")
    mask_delta = exposure_corrected_data <= detection_threshold
    exp_delta_data[mask_delta] = 0
    mean = np.sum(exp_delta_data)/reduce(mul, exp_delta_data.shape)
    mode = 0.9**2/(np.max(non_zero_elements)* pixel_area)
    alpha = 2/(mean/mode-1)+1
    q = mode * (alpha +1)
    print(f'alpha is {alpha} and q is {q}')
    print(f'mean is {mean} and mode is {mode}')

    ju.plot_result(mask_delta, n_rows=1, n_cols=3, adjust_figsize=False,
                   figsize=(15, 5), common_colorbar=False, output_file='../mask.png')

    ju.plot_result(exp_delta_data, logscale=True, n_rows=1, n_cols=3, adjust_figsize=False,
                   figsize=(15, 5), common_colorbar=False, output_file='../expdeltadata.png')

    exp_mean = np.mean(exposure_corrected_data)
    offset_mean = np.log(exp_mean)
    print(f'offset mean is {offset_mean}')
    # sat_min = [3e-10, 3e-10, 3e-10]

    # # for setting the value
    # maxim = [np.max(exposure_corrected_data[i]) for i in range(3)]
    # minim = [0.000001*i for i in maxim]
    # for i in range(3):
    #     print(f"Max Bin {i}", maxim[i])

    # # maxima set by eye
    # sat_max = [1.5e-8, 0.9e-8, 0.9e-8]

    # ju.plot_rgb(exposure_corrected_data, "blib", minim, maxim, sigma=None, log=False)
    # ju.plot_rgb(exposure_corrected_data, "blib_log", log=True)

    # ju.plot_rgb(exp_delta_data, "exp_delta", minim, maxim, sigma=None, log=False)
    # ju.plot_rgb(exp_delta_data, "exp_delta_log", log=True)

    # maxim = [0.001*np.max(summed_data[i]) for i in range(3)]
    # minim = [0.0001*i for i in maxim]
    # for i in range(3):
    #     print(f"Max Bin {i}", maxim[i])
    # sat_min = [0.01, 0.01, 0.01]
    # sat_max = [0.3e3, 0.1e3, 0.4e2]
    # ju.plot_rgb(summed_data, "data", minim, maxim, log=False)
    # ju.plot_rgb(summed_data, "data_log", log=True)
    # ju.plot_result(exposures[0], logscale=True, n_cols=3, adjust_figsize=True, common_colorbar=True)

