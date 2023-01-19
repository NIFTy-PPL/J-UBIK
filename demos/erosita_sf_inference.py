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
hyperparamerter_search = False
if __name__ == "__main__":
    config_filename = "eROSITA_config.yaml"
    try:
        cfg = xu.get_cfg(config_filename)
    except:
        cfg = xu.get_cfg('demos/' + config_filename)
    fov = cfg['telescope']['field_of_view']
    rebin = math.floor(20 * fov // cfg['grid']['npix'])

    # File Location
    file_info = cfg['files']
    obs_path = file_info['obs_path']
    input_filenames = file_info['input']
    output_filename = file_info['output']
    exposure_filename = file_info['exposure']
    observation_instance = ErositaObservation(input_filenames, output_filename, obs_path)
    sky_model = ErositaSky(config_filename)
    point_sources, diffuse, sky = sky_model.create_sky_model()

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
    psf_file = xu.eROSITA_PSF("../psf_info/tm1/bcf/tm1_2dpsf_190219v05.fits")  # fixme: load from config


    def get_lower_radec_from_pointing(center, domain, return_shift=False):
        shift = np.array(domain.shape) / 2 * np.array(domain.distances)
        if return_shift:
            return shift
        return center - shift


    shift = np.array(sky_model.position_space.shape) / 2 * np.array(sky_model.position_space.distances)
    psf_function = psf_file.psf_func_on_domain('3000', center, sky_model.position_space,
                                               get_lower_radec_from_pointing(center, sky_model.position_space))

    psf_kernel = psf_function(*get_lower_radec_from_pointing(center, sky_model.position_space, return_shift=True))
    # p = ift.Plot()
    # p.add(ift.makeField(sky_model.position_space, psf_kernel), norm=colors.SymLogNorm(linthresh=10e-8))
    # p.output()

    convolved_sky = xu.convolve_field_operator(psf_kernel, sky)
    convolved_ps = xu.convolve_field_operator(psf_kernel, point_sources)
    convolved_diffuse = xu.convolve_field_operator(psf_kernel, diffuse)

    # Exposure
    exposure = observation_instance.load_fits_data(exposure_filename)[0].data
    exposure_field = ift.makeField(sky_model.position_space, exposure)
    padded_exposure_field = sky_model.pad(exposure_field)
    exposure_op = ift.makeOp(padded_exposure_field)

    # Mask
    mask = xu.get_mask_operator(exposure_field)

    # Response
    R = mask @ sky_model.pad.adjoint @ exposure_op

    # Data
    data = observation_instance.load_fits_data(output_filename)[0].data
    data = ift.makeField(sky_model.position_space, data)
    padded_data = sky_model.pad(data)
    masked_data = mask(data)

    if mockrun:
        ift.random.push_sseq_from_seed(cfg['seed'])

        def get_data_realization(op, data=True):
            mock_position = ift.from_random(op.domain)
            resp = sky_model.pad.adjoint @ exposure_op
            res = op(mock_position)
            if data:
                res = resp(op(mock_position))
                res = ift.random.current_rng().poisson(res.val.astype(np.float64))
                res = ift.makeField(sky_model.pad.adjoint.target, res)
            return res


        if plot_info['enabled']:
            if hyperparamerter_search:
                for alpha in list(np.linspace(0.5, 5, 20)):
                    for q in [5 * 1e-5, 1e-5, 5 * 1e-4, 1e-4, 5 * 1e-3]:
                        sky_model_new = ErositaSky(config_filename, alpha=alpha, q=q)
                        sky, point_sources, diffuse = sky_model_new.create_sky_model()
                        convolved_sky = xu.convolve_field_operator(psf_kernel, sky)
                        convolved_ps = xu.convolve_field_operator(psf_kernel, point_sources)
                        convolved_diffuse = xu.convolve_field_operator(psf_kernel, diffuse)

                        # Get mock data
                        mock_data = get_data_realization(convolved_sky)
                        mock_ps_data = get_data_realization(convolved_ps)
                        mock_diffuse_data = get_data_realization(convolved_diffuse)

                        # Get mock signal
                        mock_sky = get_data_realization(convolved_sky, data=False)
                        mock_ps = get_data_realization(convolved_ps, data=False)
                        mock_diffuse = get_data_realization(convolved_diffuse, data=False)

                        print(f'Plotting  mock data for alpha = {alpha} and q = {q}.')
                        p = ift.Plot()
                        norm = colors.SymLogNorm(linthresh=10e-8)
                        p.add(mock_ps, title='point sources response', norm=norm)
                        p.add(mock_diffuse, title='diffuse component response', norm=norm)
                        p.add(mock_sky, title='sky', norm=norm)
                        p.add(mock_ps_data, title='mock point source data', norm=norm)
                        p.add(data, title='data', norm=norm)
                        p.add(mock_data, title='mock data', norm=norm)
                        p.add(mock_diffuse_data, title='mock diffuse data', norm=norm)
                        p.output(nx=4, name=f'mock_data_a{alpha}_q{q}.png')

            # Get mock data
            mock_data = get_data_realization(convolved_sky)
            mock_ps_data = get_data_realization(convolved_ps)
            mock_diffuse_data = get_data_realization(convolved_diffuse)

            # Get mock signal
            mock_sky = get_data_realization(convolved_sky, data=False)
            mock_ps = get_data_realization(convolved_ps, data=False)
            mock_diffuse = get_data_realization(convolved_diffuse, data=False)

            p = ift.Plot()
            norm = colors.SymLogNorm(linthresh=10e-8)
            p.add(mock_ps, title='point sources response', norm=norm)
            p.add(mock_diffuse, title='diffuse component response', norm=norm)
            p.add(mock_sky, title='sky', norm=norm)
            p.add(mock_ps_data, title='mock point source data', norm=norm)
            p.add(data, title='data', norm=norm)
            p.add(mock_data, title='mock data', norm=norm)
            p.add(mock_diffuse_data, title='mock diffuse data', norm=norm)
            p.output(nx=4, name=f'mock_data.png')



    # Print Exposure norm
    # norm = xu.get_norm(exposure, data)
    # print(norm)

    # Set up likelihood
    if mockrun:
        log_likelihood = ift.PoissonianEnergy(masked_data) @ R @ convolved_sky
    else:
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
                        export_operator_outputs=operators_to_plot, inspect_callback=plot, resume=True)
    else:
        # MGVI
        ift.optimize_kl(log_likelihood, minimization_config['total_iterations'], minimization_config['n_samples'],
                        minimizer, ic_sampling, None, export_operator_outputs=operators_to_plot,
                        output_directory=output_directory, inspect_callback=plot, resume=True)
