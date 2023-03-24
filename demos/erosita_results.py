import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import nifty8 as ift
from matplotlib.colors import LogNorm

import xubik0 as xu


def get_rel_unc(mean, std):
    assert mean.domain == std.domain
    domain = mean.domain
    mean, std = mean.val, std.val
    res = np.zeros(mean.shape)
    mask = mean != 0
    res[mask] = std[mask] / mean[mask]
    res[~mask] = np.nan
    return ift.makeField(domain, res)


from  matplotlib.colors import LinearSegmentedColormap
redmap = LinearSegmentedColormap.from_list('kr', ["k", "darkred", "sandybrown"], N=256)
greenmap = LinearSegmentedColormap.from_list('kg', ["k", "g", "palegreen"], N=256)
bluemap = LinearSegmentedColormap.from_list('kb', ["k", "b", "paleturquoise"], N=256)

COLOR_DICT = {'red':{'path':'red','config':'_red', 'cmap':redmap},
              'green':{'path':'green','config':'_green', 'cmap':greenmap},
              'blue' :{'path':'blue', 'config':'_blue','cmap':bluemap}}

if __name__ == '__main__':
    col = 'red'
    # Paths -Set by user
    output_path = f'final_results_{col}/'
    xu.create_output_directory(output_path)
    reconstruction_path_list = [f"results/LMC_{COLOR_DICT[col]['path']}/",]#["results/LMC_red/", "results/LMC_first_blue_llzm_notrans/",
                               # "results/LMC_low_flex/"]  # FIXME filepath
    if len(reconstruction_path_list) > 3:
        raise NotImplementedError
    diagnostics_path_list = [r_path + "diagnostics/" for r_path in reconstruction_path_list]
    config_filename = f"eROSITA_config{COLOR_DICT[col]['config']}.yaml"
    sl_path_base_list = [r_path + "pickle/last" for r_path in
                         reconstruction_path_list]  # NIFTy dependency
    data_base = "data.pkl"
    mock_data_base = "mock_data_sky.pkl"
    exposure_base = "exposure.pkl"

    cfg_filename_list = [r_path + config_filename for r_path in reconstruction_path_list]
    plottable_field_list = []

    mask_plots = True

    for i in range(len(cfg_filename_list)):
        # Config
        cfg = xu.get_cfg(cfg_filename_list[i])
        file_info = cfg['files']
        obs_path = file_info['obs_path']
        exposure_filename = file_info['exposure']

        # Telescope Info
        tel_info = cfg['telescope']  # FIXME
        tm_ids = tel_info['tm_ids']

        sky_model = xu.SkyModel(cfg_filename_list[i])
        sky_dict = sky_model.create_sky_model()
        padder = sky_model.pad

        joint_mask_filename = "joint_mask.pickle"
        joint_mask_path = os.path.join(output_path, joint_mask_filename)

        if os.path.exists(joint_mask_path):
            with open(joint_mask_path, 'rb') as f:
                joint_mask_field = pickle.load(f)

        else:
            response_dict = xu.load_erosita_response(cfg_filename_list[i], 
                                                     diagnostics_path_list[i])

            # Create joint mask
            joint_mask_field = None
            joint_mask = np.ones(sky_model.position_space.shape)
            for tm_id in tm_ids:
                tm_directory = xu.create_output_directory(
                    os.path.join(diagnostics_path_list[i], f'tm{tm_id}/'))
                tm_key = f'tm_{tm_id}'

                if exposure_base is not None:
                    exposure_path = tm_directory + f"tm{tm_id}_{exposure_base}"
                    with open(exposure_path, "rb") as f:
                        exposure_field = pickle.load(f)

                joint_mask[exposure_field.val != 0] = 0
            joint_mask_field = ift.makeField(exposure_field.domain, joint_mask)

            with open(joint_mask_path, 'wb') as f:
                pickle.dump(joint_mask_field, f)

        joint_mask = ift.MaskOperator(joint_mask_field)
        joint_mask = joint_mask.adjoint @ joint_mask

        if mask_plots:
            masked_sky_dict = sky_dict.copy()
            pspec = masked_sky_dict.pop('pspec')
            for key, val in masked_sky_dict.items():
                masked_sky_dict[key] = joint_mask @ padder.adjoint @ val

            masked_sky_dict['pspec'] = pspec
        else:
            masked_sky_dict = None

        # Get result fields to be plotted
        plottable_operators = sky_dict.copy() if masked_sky_dict is None else masked_sky_dict.copy()
        power_spectrum = plottable_operators.pop('pspec')  # FIXME

        # Load sample list
        sample_list = ift.ResidualSampleList.load(sl_path_base_list[i])

        plottable_fields = {}
        for key, op in plottable_operators.items():
            mean, var = sample_list.sample_stat(op)
            plottable_fields[key] = {}
            plottable_fields[key]['mean'] = mean
            plottable_fields[key]['std'] = var.sqrt()
        plottable_field_list.append(plottable_fields)

    # Plot results
    outname_base = output_path + "final_res_{}_{}.png"

    if len(plottable_field_list) == 1:
        for key, stat in plottable_field_list[0].items():
            args = {'norm': LogNorm(vmin=8E-6, vmax=1E-3), 
                    'cmap': COLOR_DICT[col]['cmap'],
                    'title': "Reconstruction"}
            xu.plot_result(stat['mean'], outname_base.format(key, 'mean'), 
                           **args)
            args = {'norm':LogNorm(vmin=5E-6, vmax=1E-3), 
                    'cmap': 'cividis',
                    'title': "Absolute uncertainty"}
            xu.plot_result(stat['std'], outname_base.format(key, 'std'), **args)

            from matplotlib.ticker import FuncFormatter
            fmt=FuncFormatter(lambda x, pos: '{:.1%}'.format(x))
            args = {'vmin': 0.05, 
                    'vmax': 1., 
                    'cmap': 'cividis',
                    'title': "Relative uncertainty"}
            xu.plot_result(get_rel_unc(stat['mean'],stat['std']), 
                           outname_base.format(key, 'rel_std'), 
                           cbar_formatter=fmt, **args)
            print(f'Results saved as {outname_base.format(key, "stat")} for stat in (mean, std, rel).')

    else:
        mean = {}
        std = {}
        for key in plottable_field_list[0].keys():
            mean[key] = np.stack([dct[key]['mean'].val for dct in plottable_field_list])
            std[key] = np.stack([dct[key]['std'].val for dct in plottable_field_list])

        sky = mean['sky']
        sky = np.transpose(sky, (1, 2, 0))

        mf_domain = ift.DomainTuple.make((sky_model.position_space, ift.RGSpace(3)))
        sky_field = ift.makeField(mf_domain, sky)
        xu.save_rgb_image_to_fits(sky_field, output_path + "skyRGB_lin.fits", True, True)

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