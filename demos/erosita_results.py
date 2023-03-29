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


def convert_field_to_RGB(field, norm=None, sat=None):
    if norm is None:
        norm = [np.log, np.log10, np.log10]
    if sat is None:
        sat = [1.75, 1.4, 1.3]
    arr = field.val
    res = []
    for i in range(3):
        sub_array = arr[:, :, i]
        color_norm = norm[i]
        r = np.zeros_like(sub_array)
        mask = sub_array != 0
        if norm is not None:
            r[mask] = color_norm(sub_array[mask]) if color_norm is not None else sub_array[mask]
            min = np.min(r[mask])
            max = np.max(r[mask])
            r[mask] -= min
            r[mask] /= (max - min)
            r[mask] *= sat[i]
            r[mask] = r[mask] * 255.0
            r[~mask] = 0
        res.append(r)
    res = np.array(res, dtype='int')
    res = np.transpose(res, (1, 2, 0))
    return res


from matplotlib.colors import LinearSegmentedColormap
redmap = LinearSegmentedColormap.from_list('kr', ["k", "darkred", "sandybrown"], N=256)
greenmap = LinearSegmentedColormap.from_list('kg', ["k", "g", "palegreen"], N=256)
bluemap = LinearSegmentedColormap.from_list('kb', ["k", "b", "paleturquoise"], N=256)

COLOR_DICT = {'red': {'path': 'red', 'config': '_red', 'cmap': redmap},
              'green': {'path': 'green', 'config': '_green', 'cmap': greenmap},
              'blue': {'path': 'blue', 'config': '_blue', 'cmap': bluemap}}

if __name__ == '__main__':
    colors = ['red', 'green', 'blue']
    if len(colors) > 3 or len(colors) < 1:
        raise NotImplementedError

    # Base path names for data and exp
    data_base = "data.pkl"
    mock_data_base = "mock_data_sky.pkl"
    exposure_base = "exposure.pkl"

    output_path_base = 'final_results_{}/'
    output_paths = []
    # reconstruction_paths = []
    diagnostic_paths = []
    config_paths = []
    samples_paths = []

    # Paths -Set by user
    for col in colors:
        output_path = output_path_base.format(col)
        xu.create_output_directory(output_path)
        output_paths.append(output_path)
        reconstruction_path = f"results/LMC_{COLOR_DICT[col]['path']}/"
        # reconstruction_paths.append(reconstruction_path)
        diagnostic_paths.append(reconstruction_path + "diagnostics/")
        config_paths.append(reconstruction_path + f"eROSITA_config{COLOR_DICT[col]['config']}.yaml")
        samples_paths.append(reconstruction_path + "pickle/last")

    plottable_field_list = []
    mask_plots = True

    for i in range(len(colors)):
        # Config
        cfg = xu.get_cfg(config_paths[i])
        file_info = cfg['files']
        obs_path = file_info['obs_path']
        exposure_filename = file_info['exposure']

        # Telescope Info
        tel_info = cfg['telescope']  # FIXME
        tm_ids = tel_info['tm_ids']

        sky_model = xu.SkyModel(config_paths[i])
        sky_dict = sky_model.create_sky_model()
        padder = sky_model.pad

        joint_mask_filename = "joint_mask.pickle"
        joint_mask_path = os.path.join(output_paths[i], joint_mask_filename)

        if os.path.exists(joint_mask_path):
            with open(joint_mask_path, 'rb') as f:
                joint_mask_field = pickle.load(f)

        else:
            response_dict = xu.load_erosita_response(config_paths[i], diagnostic_paths[i])

            # Create joint mask
            joint_mask_field = None
            joint_mask = np.ones(sky_model.position_space.shape)
            for tm_id in tm_ids:
                tm_directory = xu.create_output_directory(
                    os.path.join(diagnostic_paths[i], f'tm{tm_id}/'))
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
        sample_list = ift.ResidualSampleList.load(samples_paths[i])

        plottable_fields = {}
        for key, op in plottable_operators.items():
            mean, var = sample_list.sample_stat(op)
            plottable_fields[key] = {}
            plottable_fields[key]['mean'] = mean
            plottable_fields[key]['std'] = var.sqrt()
        plottable_field_list.append(plottable_fields)

    # Plot results
    if len(plottable_field_list) == 1:
        outname_base = output_paths[0] + "final_res_{}_{}.png"
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
            args = {'vmin': 1e-7,
                    'vmax': 1., 
                    'cmap': 'cividis',
                    'title': "Relative uncertainty"}
            xu.plot_result(get_rel_unc(stat['mean'], stat['std']),
                           outname_base.format(key, 'rel_std'), 
                           cbar_formatter=fmt, **args)
            print(f'Results saved as {outname_base.format(key, "stat")} for stat in'
                  f' {{mean, std, rel_std}}.')

    else:
        # Create RGB image
        output_path = output_path_base.format("RGB")
        mean = {}
        std = {}
        for key in plottable_field_list[0].keys():
            mean[key] = np.stack([dct[key]['mean'].val for dct in plottable_field_list])
            std[key] = np.stack([dct[key]['std'].val for dct in plottable_field_list])

        sky = mean['sky']
        sky = np.transpose(sky, (1, 2, 0))

        mf_domain = ift.DomainTuple.make((sky_model.position_space, ift.RGSpace(3)))
        sky_field = ift.makeField(mf_domain, sky)

        xu.create_output_directory(output_path)
        xu.save_rgb_image_to_fits(sky_field, output_path + "skyRGB_lin.fits", True, True)

        im = plt.imshow(convert_field_to_RGB(sky_field), origin="lower")
        rgb_filename = output_path + "sky_rgb.png"
        plt.savefig(rgb_filename, dpi=300)
        print(f"RGB image saved as {rgb_filename}.")
        print()

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
