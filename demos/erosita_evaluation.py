import argparse
from os.path import join
import pickle

import jax
import numpy as np
from jax import tree_map

import jubik0 as ju
import nifty8.re as jft

# Parser Setup
parser = argparse.ArgumentParser()
parser.add_argument('evaluation_config', type=str,
                    help="Config file (.yaml) for eROSITA inference.",
                    nargs='?', const=1, default="eROSITA_evaluation_config.yaml")
args = parser.parse_args()


def get_sub_dictionary(dictionary, keys):
    '''
    Returns a dictionary containing only the keys specified in keys
    '''
    return {key: dictionary[key] for key in keys if key in dictionary}


def run_evaluation_function(function, samples, operators_dict, diagnostics_path, response_dict,
                            masked_data, gt_dict, kwargs):
    '''
    Runs evaluation-specific functions on samples and saves results in diagnostics_path.
    '''
    reference = None
    match function:
        case 'noise_weighted_residuals':
            function = ju.compute_noise_weighted_residuals
            reference = masked_data.copy()
        case 'uncertainty_weighted_residuals':
            reference = gt_dict
            if reference is {}:
                return
            function = ju.compute_uncertainty_weighted_residuals
        case 'uncertainty_weighted_mean':
            function = ju.compute_uncertainty_weighted_residuals
            reference = None
        case 'rel_lambda_2D_histogram' | 'lambda_2D_histogram'| \
             'signal_space_2D_histogram' | 'rel_signal_space_2D_histogram':
            reference = gt_dict
            if reference is {}:
                return
            function = ju.plot_2d_gt_vs_rec_histogram
            if "offset" not in kwargs:
                kwargs["offset"] = 1.e-10
        case _:
            raise ValueError(f'Function {function} is not supported.')
    function(samples, operators_dict, diagnostics_path, response_dict, reference, **kwargs)


def run_evaluation_from_config(samples, config, op_dict, diagnostics_path, response_dict,
                               masked_data, gt_dict):
    '''
    Runs evaluation-specific functions on samples and saves results in diagnostics_path from
    config file.
    '''
    diagnostics_cfg = config['diagnostics']
    for k, v in diagnostics_cfg.items():
        operator_dict = get_sub_dictionary(op_dict, v['output_operators_keys'])
        ground_truth_dict = get_sub_dictionary(gt_dict, v['output_operators_keys'])
        run_evaluation_function(k, samples, operator_dict, diagnostics_path, response_dict,
                                masked_data, ground_truth_dict, v['kwargs'])


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
    config_path = join(reconstruction_path, eval_cfg['config_name'])
    minimization_output_file = join(reconstruction_path, eval_cfg['minimization_output_file'])
    diagnostics_path = join(reconstruction_path, "diagnostics")

    cfg = ju.get_config(config_path)
    tel_info = cfg['telescope']
    file_info = cfg['files']

    # Load response
    response_dict = ju.build_erosita_response_from_config(config_path)

    # Load sky models
    sky_model = ju.SkyModel(config_path)
    sky = sky_model.create_sky_model()
    sky_dict = sky_model.sky_model_to_dict()

    # Load telescope information
    mask_adj = jax.linear_transpose(response_dict['mask'],
                                    np.zeros((len(tel_info['tm_ids']),) + sky.target.shape))
    response_dict['mask_adj'] = mask_adj
    mask_func = response_dict['mask']

    # Load data and ground truth
    gt_dict = {}
    masked_data = ju.load_masked_data_from_config(config_path)
    try:
        pos = ju.load_mock_position_from_config(config_path)
        for key, comp in sky_dict.items():
            gt_dict[key] = comp(pos)
    except:
        print("Ground truth not available."
              "Ground-truth-dependent metrics (e.g. uncertainty-weighted residuals) "
              "will not be calculated.")

    masked_data = tree_map(lambda x: np.array(x, dtype=np.float64),
                           masked_data)  # convert to float64

    with open(minimization_output_file, "rb") as file:
        samples, _ = pickle.load(file)

    # Make a fake samples object for MAP case
    if not samples:
        samples = {key: np.array([val]) for key, val in samples.pos.tree.items()}
        samples = jft.Samples(pos=None, samples=jft.Vector(samples))

    run_evaluation_from_config(samples, eval_cfg, sky_dict, diagnostics_path, response_dict,
                               masked_data, gt_dict)
