import argparse
import math
import os
import sys

import numpy as np
from matplotlib import colors
import nifty8 as ift
import xubik0 as xu
from demos.sky_model import ErositaSky

from src.library.plot import plot_sample_and_stats, create_output_directory

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from src.library.erosita_observation import ErositaObservation

mockrun = True
if __name__ == "__main__":
    config_filename = "demos/eROSITA_config.yaml"
    cfg = xu.get_cfg(config_filename)
    fov = cfg['telescope']['field_of_view']
    rebin = math.floor(20 * fov//cfg['grid']['npix'])

    # File Location
    file_info = cfg['files']
    obs_path = file_info['obs_path']
    input_filenames = file_info['input']
    output_filename = file_info['output']
    exposure_filename = file_info['exposure']
    observation_instance = ErositaObservation(input_filenames, output_filename, obs_path)
    point_sources, diffuse, sky = ErositaSky(config_filename).create_sky_model()

    # Grid Info
    grid_info = cfg['grid']
    e_min = grid_info['energy_bin']['e_min']
    e_max = grid_info['energy_bin']['e_max']
    npix = grid_info['npix']

    # Telescope Info
    tel_info = cfg['telescope']
    tm_id = tel_info['tm_id']

    log = 'Output file {} already exists and is not regenerated. ' \
          'If the observations parameters shall be changed please delete or rename the current output file.'

    if not os.path.exists(os.path.join(obs_path, output_filename)):
        observation = observation_instance.get_data(emin=e_min, emax=e_max, image=True, rebin=rebin,
                                                    size=npix, pattern=tel_info['pattern'],
                                                    telid=tm_id)  # FIXME: exchange rebin by fov? 80 = 4arcsec
    else:
        print(log.format(os.path.join(obs_path, output_filename)))

    observation_instance = ErositaObservation(output_filename, output_filename, obs_path)

    # Exposure
    if not os.path.exists(os.path.join(obs_path, exposure_filename)):
        observation_instance.get_exposure_maps(output_filename, e_min, e_max, mergedmaps=exposure_filename)

    else:
        print(log.format(os.path.join(obs_path, output_filename)))

    # Plotting
    plot_info = cfg['plotting']
    if plot_info['enabled']:
        observation_instance.plot_fits_data(output_filename,
                                            os.path.splitext(output_filename)[0],
                                            slice=plot_info['slice'],
                                            dpi=plot_info['dpi'])
        observation_instance.plot_fits_data(exposure_filename,
                                            f'{os.path.splitext(exposure_filename)[0]}.png',
                                            slice=plot_info['slice'],
                                            dpi=plot_info['dpi'])

    # PSF
    center = observation_instance.get_center_coordinates(output_filename)

    # Exposure
    exposure = observation_instance.load_fits_data(exposure_filename)[0].data
    exposure = ift.makeField(sky.target, exposure)
    exposure_op = ift.makeOp(exposure)

    # Mask
    mask = xu.get_mask_operator(exposure)

    # Response
    R = mask @ exposure_op

    # Data
    data = observation_instance.load_fits_data(output_filename)[0].data
    data = ift.makeField(sky.target, data)
    masked_data = mask(data)

    if mockrun:
        mock_position = ift.from_random(sky.domain)
        mock_sky = sky(mock_position)
        mock_data = np.random.poisson(exposure_op(mock_sky).val.astype(np.float64))
        mock_data = ift.Field.from_raw(sky.target, mock_data)

        if plot_info['enabled']:
            p = ift.Plot()
            p.add(data, title='data', norm=colors.SymLogNorm(linthresh=10e-5))
            p.add(mock_data, title='mock_data', norm=colors.SymLogNorm(linthresh=10e-5))
            p.output(nx=2)

        masked_data = mask(mock_data)


    # Set up likelihood
    log_likelihood = ift.PoissonianEnergy(masked_data) @ R @ sky

    # Load minimization config
    minimization_config = cfg['minimization']

    # Minimizers
    ic_newton = ift.AbsDeltaEnergyController(**minimization_config['ic_newton'])
    ic_sampling = ift.AbsDeltaEnergyController(**minimization_config['ic_sampling'])
    ic_sampling_nl = ift.AbsDeltaEnergyController(**minimization_config['ic_sampling_nl'])
    minimizer = ift.NewtonCG(ic_newton)
    minimizer_sampling = ift.NewtonCG(ic_sampling_nl)

    # Prepare results
    operators_to_plot = {'reconstruction': sky, 'point_sources': point_sources, 'diffuse_component': diffuse}

    output_directory = create_output_directory("retreat_first_reconstruction")

    plot = lambda x, y: plot_sample_and_stats(output_directory, operators_to_plot, x, y,
                                              plotting_kwargs={'norm': colors.SymLogNorm(linthresh=10e-1)})

    if minimization_config['geovi']:
        # geoVI
        ift.optimize_kl(log_likelihood, minimization_config['total_iterations'], minimization_config['n_samples'],
                        minimizer, ic_sampling, minimizer_sampling, output_directory=output_directory,
                        export_operator_outputs=operators_to_plot, inspect_callback=plot)
    else:
        # MGVI
        ift.optimize_kl(log_likelihood, minimization_config['total_iterations'], minimization_config['n_samples'],
                        minimizer, ic_sampling, None, export_operator_outputs=operators_to_plot,
                        output_directory=output_directory, inspect_callback=plot)


