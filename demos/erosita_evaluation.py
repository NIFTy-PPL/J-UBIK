import argparse
from os.path import join
import pickle

import jubik0 as ju

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
            - Histograms for noise-weighted residuals for each dataset and as average
            - Sample averaged 2D histogram of distances in data space
                                                                       
        Non-Module specific diagnostics:
            - uncertainty weighted mean 
            Only for mock:
            - uncertainty-weighted residuals
            - Sample averaged 2D histogram of distances in signal space
            
    """
    # Set paths
    eval_cfg = ju.get_config(args.evaluation_config)
    reconstruction_path = eval_cfg['results_dir']
    iteration = eval_cfg['iteration']
    config_path = join(reconstruction_path, eval_cfg['config_name'])
    sl_path_base = join(reconstruction_path, f'samples_{iteration}')
    state_path_base = join(reconstruction_path, f'state_{iteration}')
    diagnostics_path = join(reconstruction_path, "diagnostics")

    cfg = ju.get_config(config_path)
    tel_info = cfg['telescope']
    file_info = cfg['files']

    response_dict = ju.build_erosita_response_from_config(config_path)

    mask_func = response_dict['mask']
    response_func = response_dict['R']

    # Signal space UWRs
    uwr_cfg = eval_cfg['uwr']
    if not cfg['mock'] and uwr_cfg is not None:
        raise Warning('Not able to calculate the UWRs in signal space as'
                      'the considered run is not a mock run.')

    elif cfg['mock'] and uwr_cfg is not None:
        output_operator_keys = uwr_cfg['output_operators_keys']
        gt_dict = {}
        for key in output_operator_keys:
            with open(join(reconstruction_path, f'{key}_gt.pkl'), 'rb') as file:
                gt_dict[key] = pickle.load(file)

        ju.get_diagnostics_from_file(ju.compute_uncertainty_weighted_residuals,
                                     diagnostics_path,
                                     sl_path_base,
                                     state_path_base,
                                     config_path,
                                     output_operator_keys,
                                     reference_dict=gt_dict,
                                     output_dir_base=uwr_cfg['base_filename'],
                                     mask=uwr_cfg['mask'] if 'mask' in uwr_cfg else False,
                                     abs=uwr_cfg['abs'] if 'abs' in uwr_cfg else False,
                                     n_bins=uwr_cfg['n_bins'] if 'n_bins' in uwr_cfg else None,
                                     range=uwr_cfg['range'] if 'range' in uwr_cfg else None,
                                     log=uwr_cfg['log'] if 'log' in uwr_cfg else True,
                                     plot_kwargs=uwr_cfg['plot_kwargs']
                                     if 'plot_kwargs' in uwr_cfg else None)
    # Signal space UWM
    uwm_cfg = eval_cfg['uwm']
    if uwm_cfg is not None:
        output_operator_keys = uwm_cfg['output_operators_keys']
        ju.get_diagnostics_from_file(ju.compute_uncertainty_weighted_residuals,
                                     diagnostics_path,
                                     sl_path_base,
                                     state_path_base,
                                     config_path,
                                     output_operator_keys,
                                     output_dir_base=uwm_cfg['base_filename'],
                                     mask=uwm_cfg['mask'] if 'mask' in uwm_cfg else False,
                                     log=uwm_cfg['log'] if 'log' in uwm_cfg else True,
                                     plot_kwargs=uwm_cfg['plot_kwargs'])

    # NWR
    nwr_cfg = eval_cfg['nwr']
    if nwr_cfg is not None:
        output_operator_keys = nwr_cfg['output_operators_keys']
        # Load data
        if cfg['mock']:
            masked_data = ju.load_masked_data_from_pickle(join(file_info['res_dir'],
                                                           'mock_data_dict.pkl'))
        else:
            masked_data = ju.load_erosita_masked_data(file_info, tel_info, mask_func)

        ju.get_diagnostics_from_file(ju.compute_noise_weighted_residuals,
                                     diagnostics_path,
                                     sl_path_base,
                                     state_path_base,
                                     config_path,
                                     output_operator_keys,
                                     response_func=response_func,
                                     reference_dict=masked_data,
                                     output_dir_base=nwr_cfg['base_filename'],
                                     n_bins=nwr_cfg['n_bins'] if 'n_bins' in nwr_cfg else None,
                                     plot_kwargs=nwr_cfg['plot_kwargs'])
    if cfg['mock']:
        # 2D histograms in signal space
        sky_2D_hist_cfg = eval_cfg['sky_2D_hist']
        if sky_2D_hist_cfg is not None:
            output_operator_keys = sky_2D_hist_cfg['output_operators_keys']
            gt_dict = {}
            for key in output_operator_keys:
                with open(join(reconstruction_path, f'{key}_gt.pkl'), 'rb') as file:
                    gt_dict[key] = pickle.load(file)
            ju.get_diagnostics_from_file(ju.plot_2d_gt_vs_rec_histogram,
                                         diagnostics_path,
                                         sl_path_base,
                                         state_path_base,
                                         config_path,
                                         output_operator_keys,
                                         response_func=None,
                                         reference_dict=gt_dict,
                                         output_dir_base=sky_2D_hist_cfg['base_filename'],
                                         plot_kwargs=sky_2D_hist_cfg['plot_kwargs'],
                                         type=sky_2D_hist_cfg['type'] if 'type' in sky_2D_hist_cfg
                                         else 'single',
                                         relative=False)

        rel_sky_2D_hist_cfg = eval_cfg['rel_sky_2D_hist']
        if rel_sky_2D_hist_cfg is not None:
            output_operator_keys = sky_2D_hist_cfg['output_operators_keys']
            gt_dict = {}
            for key in output_operator_keys:
                with open(join(reconstruction_path, f'{key}_gt.pkl'), 'rb') as file:
                    gt_dict[key] = pickle.load(file)
            ju.get_diagnostics_from_file(ju.plot_2d_gt_vs_rec_histogram,
                                         diagnostics_path,
                                         sl_path_base,
                                         state_path_base,
                                         config_path,
                                         output_operator_keys,
                                         response_func=None,
                                         reference_dict=gt_dict,
                                         output_dir_base=rel_sky_2D_hist_cfg['base_filename'],
                                         plot_kwargs=rel_sky_2D_hist_cfg['plot_kwargs'],
                                         type=rel_sky_2D_hist_cfg['type'] if 'type'
                                                                             in rel_sky_2D_hist_cfg
                                         else 'single',
                                         relative=True)

        # 2D histograms in data space
        lambda_2D_hist_cfg = eval_cfg['lambda_2D_hist']
        if lambda_2D_hist_cfg is not None:
            output_operator_keys = lambda_2D_hist_cfg['output_operators_keys']
            gt_dict = {}
            for key in output_operator_keys:
                with open(join(reconstruction_path, f'{key}_gt.pkl'), 'rb') as file:
                    gt_dict[key] = pickle.load(file)
            ju.get_diagnostics_from_file(ju.plot_2d_gt_vs_rec_histogram,
                                         diagnostics_path,
                                         sl_path_base,
                                         state_path_base,
                                         config_path,
                                         output_operator_keys,
                                         response_func=response_func,
                                         reference_dict=gt_dict,
                                         output_dir_base=lambda_2D_hist_cfg['base_filename'],
                                         plot_kwargs=lambda_2D_hist_cfg['plot_kwargs'],
                                         type=lambda_2D_hist_cfg['type'] if 'type'
                                                                            in lambda_2D_hist_cfg
                                         else 'single',
                                         relative=False)

        rel_lambda_2D_hist_cfg = eval_cfg['rel_lambda_2D_hist']
        if rel_lambda_2D_hist_cfg is not None:
            output_operator_keys = rel_lambda_2D_hist_cfg['output_operators_keys']
            gt_dict = {}
            for key in output_operator_keys:
                with open(join(reconstruction_path, f'{key}_gt.pkl'), 'rb') as file:
                    gt_dict[key] = pickle.load(file)
            ju.get_diagnostics_from_file(ju.plot_2d_gt_vs_rec_histogram,
                                         diagnostics_path,
                                         sl_path_base,
                                         state_path_base,
                                         config_path,
                                         output_operator_keys,
                                         response_func=response_func,
                                         reference_dict=gt_dict,
                                         output_dir_base=rel_lambda_2D_hist_cfg['base_filename'],
                                         plot_kwargs=rel_lambda_2D_hist_cfg['plot_kwargs'],
                                         type=rel_lambda_2D_hist_cfg['type']
                                         if 'type' in rel_lambda_2D_hist_cfg else 'single',
                                         relative=True)


