import os.path

import numpy as np
import pickle

import nifty8 as ift
import xubik0 as xu

if __name__ == "__main__":
    """This is the postprocessing pipeline for the eROSITA reconstruction:
    The uncertainty weighted residual (UWR) as well as their distribution 
    and the uncertainty weighted mean (UWM) are calculated. For mock data
    additionally the UWR in signal space is calculated."""

    # Paths -Set by user
    reconstruction_path = "results/LMC/"  # FIXME filepath
    diagnostics_path = reconstruction_path + "diagnostics/"
    config_filename = "eROSITA_config.yaml"
    sl_path_base = reconstruction_path + "pickle/last"  # NIFTy dependency
    data_base = "data.pkl"
    mock_data_base = "mock_data_sky.pkl"
    exposure_base = "exposure.pkl"
    response_base = None  # FIXME response operator shall be loaded from path


    # Config
    config_file = reconstruction_path + config_filename
    cfg = xu.get_cfg(config_file)
    mock_run = cfg['mock']
    mock_psf = cfg['mock_psf']
    file_info = cfg['files']
    obs_path = file_info['obs_path']
    exposure_filename = file_info['exposure']

    # Telescope Info
    tel_info = cfg['telescope']
    tm_ids = tel_info['tm_ids']
    start_center = None

    # Exposure Info
    det_map = tel_info['detmap']

    # Operators
    # Sky
    sky_model = xu.SkyModel(config_file)
    sky_dict = sky_model.create_sky_model()
    sky_dict.pop('pspec')
    data_space_uwrs = {key: [] for key, value in sky_dict.items()}
    noise_weighted_residuals = {key: [] for key, value in sky_dict.items()}
    full_exposure = None
    response_dict = xu.load_erosita_response(config_file, diagnostics_path)
    for tm_id in tm_ids:
        # Path
        tm_directory = xu.create_output_directory(os.path.join(diagnostics_path, f'tm{tm_id}/'))
        for key, op in sky_dict.items():
            xu.create_output_directory(os.path.join(tm_directory, key))
        if mock_run:
            data_path = tm_directory + f"tm{tm_id}_{mock_data_base}"
        else:
            data_path = tm_directory + f"tm{tm_id}_{data_base}"

        # Load observation
        output_filename = f'{tm_id}_' + file_info['output']
        observation_instance = xu.ErositaObservation(output_filename, output_filename, obs_path)

        # Repsonse
        if response_base is not None:
            response_path = tm_directory + f"tm{tm_id}_{response_base}"
            with open(response_path, "rb") as f:
                R = pickle.load(f)
        elif exposure_base is not None:
            exposure_path = tm_directory + f"tm{tm_id}_{exposure_base}"
            print('Not able to load response from file. Generating response from config ...')
            with open(exposure_path, "rb") as f:
                exposure_field = pickle.load(f)
            if full_exposure is None:
                full_exposure =exposure_field
            else:
                full_exposure = full_exposure + exposure_field
            padded_exposure_field = sky_model.pad(exposure_field)
            exposure_op = ift.makeOp(padded_exposure_field)
            mask = xu.get_mask_operator(exposure_field)
            tm_key = f'tm_{tm_id}'
            R = response_dict[tm_key]['R']

        else:
            raise NotImplementedError
        for key, op in sky_dict.items():
            data_space_uwrs[key].append(
                xu.data_space_uwr_from_file(sl_path_base=sl_path_base, data_path=data_path,
                                            sky_op=op, response_op=R, mask_op=mask,
                                            output_dir_base=os.path.join(tm_directory,
                                                                         key, f'{tm_id}_data_space_uwr')))

            noise_weighted_residuals[key].append(
                xu.get_noise_weighted_residuals_from_file(sample_list_path=sl_path_base,
                                                          data_path=data_path,
                                                          sky_op=op, response_op=R,
                                                          mask_op=mask,
                                                          output_dir=diagnostics_path,
                                                          base_filename=f'/tm{tm_id}/{key}/{tm_id}_nwr',
                                                          abs=False,
                                                          plot_kwargs={
                                                            'title': 'Noise-weighted residuals',
                                                            # 'norm': LogNorm()
                                                          }))
            if mock_run:
                if key in ['sky', 'diffuse']:
                    levels = [10, 100, 500]
                    xlim = (0.02, 170)
                    ylim = (0.02, 170)
                    bins = 600
                    ground_truth_path = os.path.join(diagnostics_path, f'mock_{key}.pkl')
                    xu.plot_lambda_diagnostics(sl_path_base, ground_truth_path, sky_dict[key], key,
                                               os.path.join(tm_directory, f'{key}_diagnostics_path.png'),
                                               R, bins=bins, x_lim=xlim, y_lim=ylim,
                                               levels=levels)

        xu.weighted_residual_distribution(sl_path_base=sl_path_base, data_path=data_path,
                                          sky_op=sky_dict['sky'], response_op=R, mask_op=mask,
                                          output_dir_base=tm_directory + f'/{tm_id}_res_distribution',
                                          title='Uncertainty Weighted Signal residuals')
        full_mask = xu.get_mask_operator(full_exposure)

    for key, op in sky_dict.items():
        xu.signal_space_uwm_from_file(sl_path_base=sl_path_base, sky_op=op,
                                      padder=sky_model.pad,
                                      output_dir_base=diagnostics_path + f'/uwm_{key}')
        field_name_list = [f'tm{tm_id}' for tm_id in tm_ids]
        if mock_run:
            ground_truth_path = os.path.join(diagnostics_path, f'mock_{key}.pkl')
            xu.signal_space_uwr_from_file(sl_path_base=sl_path_base,
                                          ground_truth_path=ground_truth_path,
                                          sky_op=op,
                                          padder=sky_model.pad,
                                          mask_op=full_mask,
                                          output_dir_base=os.path.join(diagnostics_path,
                                                                       f'signal_space_uwr_{key}'))
            xu.signal_space_weighted_residual_distribution(sl_path_base=sl_path_base,
                                                           ground_truth_path=ground_truth_path,
                                                           sky_op=sky_dict[key],
                                                           padder=sky_model.pad,
                                                           mask_op=full_mask,
                                                           sample_diag=False,
                                                           output_dir_base=diagnostics_path + f'/res_distribution_sp_{key}',
                                                           title='Uncertainty Weighted Signal residuals')

            if key in ['sky']:
                levels = [10, 100, 500]
                xlim = (0.0000001, 0.003)
                ylim = (0.0000001, 0.003)
                bins = 600
                ground_truth_path = os.path.join(diagnostics_path, f'mock_{key}.pkl')
                xu.plot_sky_flux_diagnostics(sl_path_base, ground_truth_path, sky_dict[key], key,
                                             os.path.join(diagnostics_path, f'{key}_flux_diagnostics.png'),
                                             response_dict, bins=bins,
                                             x_lim=xlim, y_lim=ylim, levels=levels)

        xu.plot_energy_slice_overview(data_space_uwrs[key], field_name_list=field_name_list,
                                      file_name=diagnostics_path + f'{key}_data_space_uwrs_overview.png',
                                      title='data_space_uwrs',
                                      logscale=True)

        xu.plot_energy_slice_overview(noise_weighted_residuals[key], field_name_list=field_name_list,
                                      file_name=diagnostics_path + f'{key}_nwr_overview.png',
                                      title='Noise-weighted residuals', logscale=False)

