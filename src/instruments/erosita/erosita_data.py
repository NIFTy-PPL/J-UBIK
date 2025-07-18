# SPDX-License-Identifier: BSD-2-Clause
# Authors: Vincent Eberle, Matteo Guardiani, Margret Westerkamp

# Copyright(C) 2024 Max-Planck-Society

# %%

from os.path import join, splitext, exists

import nifty.re as jft
import numpy as np
from astropy.io import fits
from jax import numpy as jnp

from .erosita_observation import ErositaObservation
from ...data import create_mock_data
from ...messages import log_file_exists
from ...utils import (save_to_pickle, get_config, create_output_directory,
                      copy_config)


def create_erosita_data_from_config(
        config,
        response_dict
):
    """ Wrapper function to create masked data either from
    actual eROSITA observations or from generated mock data, as specified
    in the config given at config path. In any case the data is saved to the
    same pickle file.

    Parameters
    ----------
    config : dict
        Dictionary containing the configuration parameters.
    response_dict : dict
        Dictionary of all available response functionalities i.e. response,
        mask, psf

    """
    tel_info = config["telescope"]
    file_info = config["files"]
    grid_info = config['grid']
    plot_info = config['plotting']
    data_path = join(file_info['res_dir'], file_info['data_dict'])
    if not exists(data_path):
        if bool(file_info.get("mock_gen_config")):
            jft.logger.info(
                f'Generating new mock data in {file_info["res_dir"]}...')
            mock_prior_info = get_config(file_info["mock_gen_config"])['priors']
            _ = create_mock_data(tel_info,
                                 file_info,
                                 grid_info,
                                 mock_prior_info,
                                 plot_info,
                                 config['seed'],
                                 response_dict)
            copy_config(file_info['mock_gen_config'],
                        output_dir=file_info['res_dir'])
        else:
            jft.logger.info(
                f'Generating masked eROSITA data in {file_info["res_dir"]}...')
            mask_erosita_data_from_disk(file_info,
                                        tel_info,
                                        grid_info,
                                        response_dict['mask'])
    else:
        jft.logger.info(
            f'Data in {file_info["res_dir"]} already exists. '
            f'No data generation.')


def mask_erosita_data_from_disk(
        file_info,
        tel_info,
        grid_info,
        mask_func
):
    """ Creates and saves eROSITA masked data as pickle file from
     eROSITA processed fits-files.

    Parameters
    ----------
    file_info : dict
        Dictionary of file paths
    tel_info : dict
        Dictionary of telescope information
    grid_info : dict
        Dictionary with grid information
    mask_func : Callable
        Mask function, which takes a 3D array and makes a jft.Vector
        out of it containing a dictionary of masked arrays
    Returns
    -------
    masked_data_vector: jft.Vector
        Dictionary of masked data arrays
    """
    # load energies
    e_min = grid_info['energy_bin']['e_min']
    e_max = grid_info['energy_bin']['e_max']

    if not isinstance(e_min, list):
        raise TypeError("e_min must be a list!")

    if not isinstance(e_max, list):
        raise TypeError("e_max must be a list!")

    if len(e_min) != len(e_max):
        raise ValueError("e_min and e_max must have the same length!")

    data_list = []
    for tm_id in tel_info['tm_ids']:
        output_filenames = f'tm{tm_id}_' + file_info['output']
        output_filenames = [
            f"{output_filenames.split('.')[0]}_emin{e}_emax{E}.fits"
            for e, E in zip(e_min, e_max)]
        data = []
        for output_filename in output_filenames:
            data.append(fits.open(join(file_info['obs_path'],
                                       "processed",
                                       output_filename))[0].data)
        data = jnp.stack(jnp.array(data, dtype=int))
        data_list.append(data)
    data = jnp.stack(jnp.array(data_list, dtype=int))
    masked_data_vector = mask_func(data)
    save_to_pickle(masked_data_vector.tree,
                   join(file_info['res_dir'],
                        file_info["data_dict"]))
    return masked_data_vector


def generate_erosita_data_from_config(
        config
):
    """
    Generates eROSITA data by invoking the eSASS interface based on the
    configurations provided in a configuration dictionary.
    The function processes event files, generates exposure maps,
    and optionally plots the results as FITS images.

    Parameters
    ----------
    config : dict
        Dictionary containing the configuration parameters.
        The configuration dictionary should contain the necessary
        information about the observation settings, telescope modules,
        energy bins, file paths, and plotting options.
        For a description of the required fields in the configuration file,
        see demos/erosita_demo.py.

    Returns
    -------
    None
        The function processes the event and exposure data, saving the results
        as FITS files and optional plots in the specified directories.
        The output files will be stored in the folder specified by
        `processed_obs_folder` under the observation path.

    Notes
    -----
    - The function checks for consistency between the field of view and the
    rebinning factor.
    - It handles multiple telescope modules (TMs) and processes each
    module separately.
    - If output files already exist, they are skipped, and a log message
    is printed.
    - Exposure maps are generated for each energy bin and telescope module.
    - Optional plots can be generated if enabled in the configuration.
    """

    tel_info = config["telescope"]
    file_info = config["files"]
    grid_info = config['grid']
    plot_info = config['plotting']
    esass_image =config['esass_image']
    obs_path = file_info["obs_path"]
    sdim = grid_info['sdim']
    e_min = grid_info['energy_bin']['e_min']
    e_max = grid_info['energy_bin']['e_max']

    if not isinstance(e_min, list):
        raise TypeError("e_min must be a list!")

    if not isinstance(e_max, list):
        raise TypeError("e_max must be a list!")

    if len(e_max) != len(e_max):
        raise ValueError("e_min and e_max must have the same length!")

    rebin = tel_info["rebin"]
    rebin_check = int(np.floor(20 * tel_info['fov'] // sdim))

    if rebin != rebin_check:
        raise ValueError("rebin, which sets the angular resolution, and fov "
                         "do not match")

    processed_obs_path = create_output_directory(join(obs_path,
                                                      file_info['processed_obs_folder']))
    for tm_id in tel_info["tm_ids"]:
        output_filenames = f'tm{tm_id}_' + file_info['output']
        exposure_filenames = f'tm{tm_id}_' + file_info['exposure']
        output_filenames = [
            f"{output_filenames.split('.')[0]}_emin{e}_emax{E}.fits"
            for e, E in zip(e_min, e_max)]
        exposure_filenames = [
            f"{exposure_filenames.split('.')[0]}_emin{e}_emax{E}.fits"
            for e, E in zip(e_min, e_max)]

        for e, output_filename in enumerate(output_filenames):
            observation_instance = ErositaObservation(
                file_info["input"],
                join("processed", output_filename),
                obs_path,
                esass_image=esass_image)
            if not exists(join(processed_obs_path, output_filename)):
                _ = observation_instance.get_data(
                        pointing_center=tel_info["pointing_center"],
                        emin=e_min[e],
                        emax=e_max[e],
                        image=True,
                        rebin=rebin, #TODO: exchange rebin by fov - 80 = 4arcsec
                        size=sdim,
                        pattern=tel_info['pattern'],
                        telid=tm_id)
            else:
                log_file_exists(join(processed_obs_path, output_filename))

            observation_instance = ErositaObservation(output_filename,
                                                      output_filename,
                                                      processed_obs_path,
                                                      esass_image=esass_image)

            # Exposure
            if not exists(join(processed_obs_path, exposure_filenames[e])):
                observation_instance.get_exposure_maps(
                    output_filename,
                    e_min[e],
                    e_max[e],
                    withsinglemaps=True,
                    singlemaps=[exposure_filenames[e]],
                    withdetmaps=tel_info['detmap'],
                    badpix_correction=tel_info['badpix_correction']
                )

            else:
                log_file_exists(join(processed_obs_path, exposure_filenames[e]))

            # Plotting
            if plot_info['enabled']:
                observation_instance.plot_fits_data(
                    output_filename,
                    f'{splitext(output_filename)[0]}.png',
                    slice=plot_info['slice'],
                    dpi=plot_info['dpi'])
                observation_instance.plot_fits_data(
                    exposure_filenames[e],
                    f'{splitext(exposure_filenames[e])[0]}.png',
                    slice=plot_info['slice'],
                    dpi=plot_info['dpi'])
