import argparse
import numpy as np
from os.path import join

import xubik0 as xu

# Parser Setup
parser = argparse.ArgumentParser()
parser.add_argument('evaluation_config', type=str,
                    help="Config file (.yaml) for eROSITA inference.",
                    nargs='?', const=1, default="eROSITA_evaluation_config.yaml")
args = parser.parse_args()

if __name__ == "__main__":
    """This is the postprocessing pipeline for the eROSITA reconstruction
    It calculates and plots for the following quantities if the according parameters are specified 
    in the evaluation config. 
    If no parameters are not specified the quantities are neither calculated nor plotted. 
        Module specific diagnostics:
            - noise-weighted residuals for each dataset
            - Sample averaged 2D histogram of distances in data space
                                                                       
        Non-Module specific diagnostics:
            - uncertainty weighted mean 
            Only for mock:
            - uncertainty-weighted residuals
            - Sample averaged 2D histogram of distances in signal space
            
    """
    # Set paths
    eval_cfg = xu.get_config(args.evaluation_config)
    reconstruction_path = eval_cfg['results_dir']
    config_file = join(reconstruction_path, eval_cfg['config_name'])
    sl_path_base = reconstruction_path + "pickle/last" # NIFTy dependency
    diagnostics_path = reconstruction_path + "diagnostics/"
    xu.create_output_directory(diagnostics_path)
    data_base = "data.pkl"
    mock_data_base = "mock_data_sky.pkl"
    exposure_base = "exposure.pkl"

    # Get config info
    cfg = xu.get_config(config_file)
    tel_info = cfg['telescope']

    # Create sky operators
    sky_model = xu.SkyModel(config_file)
    sky_dict = sky_model.create_sky_model()
    sky_dict.pop('pspec')

    ########## Module-specific diagnostics #########

    noise_weighted_residuals = {key: [] for key in sky_dict.keys()}
    noise_weighted_residuals_hists = {key: [] for key in sky_dict.keys()}

    # The response
    full_exposure = None
    response_dict = xu.load_erosita_response(config_file, diagnostics_path)

    # Load nwr and uwm cfg
    nwr_cfg = eval_cfg['nwr']

    for tm_id in tel_info['tm_ids']:
        # Path
        tm_directory = xu.create_output_directory(join(diagnostics_path, f'tm{tm_id}/'))
        
        if cfg['mock']:
            data_path = tm_directory + f"tm{tm_id}_{mock_data_base}"
        else:
            data_path = tm_directory + f"tm{tm_id}_{data_base}"

        tm_key = f'tm_{tm_id}'
        if full_exposure is None:
            full_exposure = response_dict[tm_key]['exposure_field']
        else:
            full_exposure = full_exposure + response_dict[tm_key]['exposure_field']
        exposure_op = response_dict[tm_key]['exposure_op']
        mask = response_dict[tm_key]['mask']
        R = response_dict[tm_key]['R']

        for key, op in sky_dict.items():
            operator_path = xu.create_output_directory(join(tm_directory, key))
            # Noise weighted residuals
            if nwr_cfg is not None:
                mask = mask if 'mask' in nwr_cfg else None
                if not nwr_cfg['mask']:
                    mask = None
                nwr_res = xu.get_noise_weighted_residuals_from_file(sample_list_path=sl_path_base,
                                                                    data_path=data_path,
                                                                    sky_op=op, response_op=R,
                                                                    mask_op=mask,
                                                                    output_dir=operator_path,
                                                                    base_filename=f'{key}_{tm_id}_{nwr_cfg["base_filename"]}',
                                                                    abs=nwr_cfg['abs'] if 'abs' in nwr_cfg else False,
                                                                    min_counts=nwr_cfg['min_counts'] if 'min_counts' in nwr_cfg else None,
                                                                    nbins=nwr_cfg['n_bins'] if 'n_bins' in nwr_cfg else None,
                                                                    range=nwr_cfg['range'] if 'range' in nwr_cfg else None,
                                                                    plot_kwargs=nwr_cfg['plot_kwargs'] if 'plot_kwargs' in nwr_cfg else None)

                if nwr_cfg['n_bins'] is not None:
                    nwr, nwr_hist, nwr_edges = nwr_res
                    xu.plot_histograms(nwr_hist, nwr_edges, operator_path + f'tm{tm_id}/{key}_tm{tm_id}_nwr_hist',
                                       logy=nwr_cfg['log_y'], title=f'Noise-weighted residuals tm {tm_id}')
                    noise_weighted_residuals_hists[key].append(nwr_hist)
                else:
                    nwr = nwr_res
                noise_weighted_residuals[key].append(nwr)

            # 2D Histograms in data space
            if cfg['mock']:
                ground_truth_path = join(diagnostics_path, f'mock_{key}.pkl')
                # Sample averaged 2D histogram of distances in data space
                lambda_2D_hist_cfg = eval_cfg['lambda_2D_hist']
                if key in ['sky', 'diffuse']:
                    if lambda_2D_hist_cfg is not None:
                        xu.plot_2d_gt_vs_rec_histogram(sl_path_base, ground_truth_path, sky_dict[key], key,
                                                       response=R @ sky_model.pad, pad=sky_model.pad,
                                                       bins=lambda_2D_hist_cfg['bins'],
                                                       output_path=join(tm_directory,
                                                                        f'{key}_{lambda_2D_hist_cfg["output_name"]}'),
                                                       x_lim=lambda_2D_hist_cfg['x_lim'],
                                                       y_lim=lambda_2D_hist_cfg['y_lim'],
                                                       x_label=lambda_2D_hist_cfg['x_label'],
                                                       y_label=lambda_2D_hist_cfg['y_label'],
                                                       dpi=lambda_2D_hist_cfg['dpi'], title=lambda_2D_hist_cfg['title'],
                                                       type=lambda_2D_hist_cfg['type'],
                                                       relative=False)
                    rel_lambda_2D_hist_cfg = eval_cfg['rel_lambda_2D_hist']
                    if rel_lambda_2D_hist_cfg is not None:
                        xu.plot_2d_gt_vs_rec_histogram(sl_path_base, ground_truth_path, sky_dict[key], key,
                                                       response=R @ sky_model.pad, pad=sky_model.pad,
                                                       bins=rel_lambda_2D_hist_cfg['bins'],
                                                       output_path=join(tm_directory,
                                                                        f'{key}_{rel_lambda_2D_hist_cfg["output_name"]}'),
                                                       x_lim=rel_lambda_2D_hist_cfg['x_lim'],
                                                       y_lim=rel_lambda_2D_hist_cfg['y_lim'],
                                                       x_label=rel_lambda_2D_hist_cfg['x_label'],
                                                       y_label=rel_lambda_2D_hist_cfg['y_label'],
                                                       dpi=rel_lambda_2D_hist_cfg['dpi'],
                                                       title=rel_lambda_2D_hist_cfg['title'],
                                                       type=rel_lambda_2D_hist_cfg['type'],
                                                       relative=True)

    ########## Non-module-specific diagnostics #########

    full_mask = xu.get_mask_operator(full_exposure)
    for key, op in sky_dict.items():
        operator_path = xu.create_output_directory(join(tm_directory, key))
        mean_hist = np.array(noise_weighted_residuals_hists[key]).mean(axis=0)  # TM-mean of NWR
        if nwr_cfg is not None and nwr_cfg['n_bins'] is not None:
            xu.plot_histograms(mean_hist, 
                               nwr_edges, 
                               join(operator_path, f'mean_{key}_nwr_hist'),
                               logy=nwr_cfg['log_y'], 
                               title=f'Module-averaged {key} noise-weighted residuals')

        field_name_list = [f'tm{tm_id}' for tm_id in tel_info['tm_ids']]

        # Uncertainty weighted mean
        uwm_cfg = eval_cfg['uwm']
        if uwm_cfg is not None:
            uwm_full_mask = None
            if 'mask' in uwm_cfg:
                uwm_full_mask = full_mask if 'mask' in uwm_cfg else None

            if "output_name" not in uwm_cfg:
                uwm_cfg["output_name"] = "uwm"
            xu.get_uwm_from_file(sl_path_base,
                                 op,
                                 mask=uwm_full_mask,
                                 padder=sky_model.pad,
                                 output_dir_base=join(operator_path,
                                                      f'{uwm_cfg["output_name"]}_{key}'),
                                 plot_kwargs=uwm_cfg['plot_kwargs'],)

        # Mock-inference additional diagnostics
        if cfg['mock']:
            ground_truth_path = join(diagnostics_path, f'mock_{key}.pkl')
            uwr_cfg = eval_cfg['uwr']  # load uncertainty-weighted residuals config

            if uwr_cfg is not None:
                uwr_full_mask = full_mask if 'mask' in uwr_cfg else None
                if not uwr_cfg['mask']:
                    uwr_full_mask = None
                uwr_filename = join(operator_path, f'/{uwr_cfg["base_filename"]}_{key}') if \
                    'base_filename' in uwr_cfg else operator_path + f'/res_distribution_sp_{key}'

                uwr_res = xu.get_uwr_from_file(sl_path_base,
                                               ground_truth_path,
                                               op,
                                               sky_model.pad,
                                               uwr_full_mask,
                                               log=uwr_cfg['log'] if 'log' in uwr_cfg else False,
                                               output_dir_base=join(operator_path,
                                                                    f'signal_space_uwr_{key}'),
                                               abs=uwr_cfg['abs'] if 'abs' in uwr_cfg else False,
                                               n_bins=uwr_cfg['n_bins'] if 'n_bins' in uwr_cfg else None,
                                               range=uwr_cfg['range'] if 'range' in uwr_cfg else None,
                                               plot_kwargs=uwr_cfg['plot_kwargs'] if 'plot_kwargs' in uwr_cfg else None,
                                               )

                if 'nbins' in uwr_cfg and uwr_cfg['n_bins'] is not None:
                    uwr, uwr_hist, uwr_edges = uwr_res
                    xu.plot_histograms(uwr_hist, uwr_edges, uwr_filename, logy=uwr_cfg['log_y'],
                                       title=uwr_cfg['title'])

            # 2D histograms in signal space
            if key in ['sky']:
                sky_2D_hist_cfg = eval_cfg['sky_2D_hist']
                if sky_2D_hist_cfg is not None:
                    xu.plot_2d_gt_vs_rec_histogram(sl_path_base, ground_truth_path, sky_dict[key], key,
                                                   response=full_mask, pad=sky_model.pad,
                                                   bins=sky_2D_hist_cfg['bins'],
                                                   output_path=join(diagnostics_path,
                                                                            f'{key}_{sky_2D_hist_cfg["output_name"]}'),
                                                   x_lim=sky_2D_hist_cfg['x_lim'], y_lim=sky_2D_hist_cfg['y_lim'],
                                                   x_label=sky_2D_hist_cfg['x_label'], y_label=sky_2D_hist_cfg['y_label'],
                                                   dpi=sky_2D_hist_cfg['dpi'], title=sky_2D_hist_cfg['title'],
                                                   type=sky_2D_hist_cfg['type'],
                                                   relative=False)

                rel_sky_2D_hist_cfg = eval_cfg['rel_sky_2D_hist']
                if rel_sky_2D_hist_cfg is not None:
                    xu.plot_2d_gt_vs_rec_histogram(sl_path_base, ground_truth_path, sky_dict[key], key,
                                                   response=full_mask, pad=sky_model.pad,
                                                   bins=rel_sky_2D_hist_cfg['bins'],
                                                   output_path=join(diagnostics_path,
                                                                            f'{key}_{rel_sky_2D_hist_cfg["output_name"]}'),
                                                   x_lim=rel_sky_2D_hist_cfg['x_lim'],
                                                   y_lim=rel_sky_2D_hist_cfg['y_lim'],
                                                   x_label=rel_sky_2D_hist_cfg['x_label'],
                                                   y_label=rel_sky_2D_hist_cfg['y_label'],
                                                   dpi=rel_sky_2D_hist_cfg['dpi'], title=rel_sky_2D_hist_cfg['title'],
                                                   type=rel_sky_2D_hist_cfg['type'],
                                                   relative=True)


