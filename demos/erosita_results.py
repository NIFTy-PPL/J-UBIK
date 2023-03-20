import os
import pickle

import numpy as np
import nifty8 as ift
import xubik0 as xu

if __name__ == '__main__':
    # Paths -Set by user
    output_path = 'final_results/'
    xu.create_output_directory(output_path)
    reconstruction_path = "results/LMC_first_blue_llzm_notrans/"  # FIXME filepath
    diagnostics_path = reconstruction_path + "diagnostics/"
    config_filename = "eROSITA_config.yaml"
    sl_path_base = reconstruction_path + "pickle/last"  # NIFTy dependency
    data_base = "data.pkl"
    mock_data_base = "mock_data_sky.pkl"
    exposure_base = "exposure.pkl"

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
    padder = sky_model.pad
    response_dict = xu.load_erosita_response(config_file, diagnostics_path)

    # Load sample list
    sample_list = ift.ResidualSampleList.load(sl_path_base)

    # Plotting settings
    join_masks = True

    if join_masks:
        # Create joint mask
        joint_mask = np.ones(sky_model.position_space.shape)
        for tm_id in tm_ids:
            tm_directory = xu.create_output_directory(os.path.join(diagnostics_path, f'tm{tm_id}/'))
            tm_key = f'tm_{tm_id}'
            if exposure_base is not None:
                exposure_path = tm_directory + f"tm{tm_id}_{exposure_base}"
                with open(exposure_path, "rb") as f:
                    exposure_field = pickle.load(f)
            joint_mask[exposure_field.val != 0] = 0
        joint_mask = ift.MaskOperator(ift.makeField(exposure_field.domain, joint_mask))

        masked_sky_dict = sky_dict.copy()
        pspec = masked_sky_dict.pop('pspec')
        for key, val in masked_sky_dict.items():
            masked_sky_dict[key] = joint_mask.adjoint @ joint_mask @ padder.adjoint @ val
        masked_sky_dict['pspec'] = pspec
    else:
        joint_mask = None
        masked_sky_dict = None

    # Get result fields to be plotted
    plottable_operators = sky_dict.copy() if masked_sky_dict is None else masked_sky_dict.copy()
    power_spectrum = plottable_operators.pop('pspec') # FIXME

    plottable_fields = {}
    for key, op in plottable_operators.items():
        mean, var = sample_list.sample_stat(op)
        plottable_fields[key] = {}
        plottable_fields[key]['mean'] = mean
        plottable_fields[key]['std'] = var.sqrt()

    # Plot results
    outname_base = output_path + "final_res_{}_{}.pdf"
    for key, stat in plottable_fields.items():
        xu.plot_result(stat['mean'], outname_base.format(key, 'mean'), logscale=True)
        xu.plot_result(stat['std'], outname_base.format(key, 'std'), logscale=True)











    # Plot power_spectrum FIXME
    # ps_mean, ps_std = sample_list.sample_stat(power_spectrum)
    # ps_std = ps_std.sqrt()
    #
    # def _get_xy(field):
    #     return field.domain[0].k_lengths, field.val
    #
    # import matplotlib.pyplot as plt
    # plt.plot(*_get_xy(ps_mean))
    # plt.plot(*_get_xy(ps_std))
    #
    # # plt.plot(*_get_xy(ps_mean + ps_std))
    # # plt.plot(*_get_xy(ps_mean - ps_std))
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.show()



