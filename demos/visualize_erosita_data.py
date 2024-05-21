import matplotlib.pyplot as plt
import numpy as np

import jubik0 as ju
from os.path import join
import astropy.io.fits as fits


if __name__ == "__main__":
    config_path = "eROSITA_config.yaml"
    config_dict = ju.get_config(config_path)
    path_to_caldb = '~/PycharmProjects/jubik/data/'
    file_info = config_dict["files"]
    grid_info = config_dict["grid"]
    tm_ids = config_dict["telescope"]["tm_ids"]
    e_min = grid_info['energy_bin']['e_min']
    e_max = grid_info['energy_bin']['e_max']
    data_path = join(file_info['obs_path'], 'processed')

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
    correct_exposures_for_effective_area = True
    if correct_exposures_for_effective_area:
        from src.library.response import calculate_erosita_effective_area
        ea = calculate_erosita_effective_area(path_to_caldb, tm_ids, e_min, e_max)
        exposures *= ea[:, :, np.newaxis, np.newaxis]

    ju.plot_result(data[0], logscale=True, n_rows=1,n_cols=3, figsize=(15, 5),
                   adjust_figsize=False, common_colorbar=False)
    ju.plot_result(exposures[0], logscale=True, n_rows=1, n_cols=3, adjust_figsize=False,
                   figsize=(15, 5), common_colorbar=False)
    ju.plot_result(np.sum(data, axis=0)/np.sum(exposures, axis=0), logscale=True, n_rows=1, n_cols=3, adjust_figsize=False,
                   figsize=(15, 5), common_colorbar=False)

    # ju.plot_result(exposures[0], logscale=True, n_cols=3, adjust_figsize=True, common_colorbar=True)
    plt.imshow(exposures[0][1] - exposures[0][2], origin='lower')
    # plt.imshow(exposures[0][1]/exposures[0][2], origin='lower')
    plt.colorbar()
    plt.show()

