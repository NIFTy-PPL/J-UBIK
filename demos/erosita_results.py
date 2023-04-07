import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import nifty8 as ift
from matplotlib.colors import LogNorm

import xubik0 as xu


from matplotlib.colors import LinearSegmentedColormap
redmap = LinearSegmentedColormap.from_list('kr', ["k", "darkred", "sandybrown"], N=256)
greenmap = LinearSegmentedColormap.from_list('kg', ["k", "g", "palegreen"], N=256)
bluemap = LinearSegmentedColormap.from_list('kb', ["k", "b", "paleturquoise"], N=256)


def build_joint_mask(sky_model, tm_ids, diagnostics_path, exposure_base):
    joint_mask = np.ones(sky_model.position_space.shape)
    for tm_id in tm_ids:
        tm_directory = xu.create_output_directory(os.path.join(diagnostics_path, f'tm{tm_id}/'))

        if exposure_base is not None:
            exposure_path = tm_directory + f"tm{tm_id}_{exposure_base}"
            with open(exposure_path, "rb") as f:
                exposure_field = pickle.load(f)

        joint_mask[exposure_field.val != 0] = 0
    joint_mask_field = ift.makeField(exposure_field.domain, joint_mask)

    mask = ift.MaskOperator(joint_mask_field)
    return mask.adjoint @ mask

# Paths set by user
COLOR_DICT = {'red': {'path': 'red', 'config': '_red', 'cmap': redmap},
              'green': {'path': 'green', 'config': '_green', 'cmap': greenmap},
              'blue': {'path': 'blue', 'config': '_blue', 'cmap': bluemap}}

if __name__ == '__main__':
    colors = ['green']

    # Base path names for data and exp
    data_base = "data.pkl"
    mock_data_base = "mock_data_sky.pkl"
    exposure_base = "exposure.pkl"

    output_path_base = 'final_results_{}/'
    output_paths = []
    diagnostic_paths = []
    config_paths = []
    samples_paths = []

    for col in colors:
        output_path = output_path_base.format(col)
        xu.create_output_directory(output_path)
        output_paths.append(output_path)
        reconstruction_path = f"results/LMC_{COLOR_DICT[col]['path']}/"
        diagnostic_paths.append(reconstruction_path + "diagnostics/")
        config_paths.append(reconstruction_path + f"eROSITA_config{COLOR_DICT[col]['config']}.yaml")
        samples_paths.append(reconstruction_path + "pickle/last")

    plottable_field_list = []
    mask_plots = True

    for i, col in enumerate(colors):
        # Config
        cfg = xu.get_config(config_paths[i])
        file_info = cfg['files']
        obs_path = file_info['obs_path']
        exposure_filename = file_info['exposure']

        # Telescope Info
        tel_info = cfg['telescope']  # FIXME
        tm_ids = tel_info['tm_ids']

        sky_model = xu.SkyModel(config_paths[i])
        sky_dict = sky_model.create_sky_model()
        padder = sky_model.pad

        joint_mask = build_joint_mask(sky_model, tm_ids, diagnostic_paths[i], exposure_base)

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
        outname_base = output_paths[i] + "final_res_{}_{}.png"
        for key, stat in plottable_field_list[i].items():
            args = {'norm': LogNorm(vmin=8E-6, vmax=1E-3),
                    'cmap': COLOR_DICT[col]['cmap'],
                    'title': "Reconstruction"}
            xu.plot_result(stat['mean'], outname_base.format(key, 'mean'),
                           **args)
            args = {'norm': LogNorm(vmin=5E-6, vmax=1E-3),
                    'cmap': 'cividis',
                    'title': "Absolute uncertainty"}
            xu.plot_result(stat['std'], outname_base.format(key, 'std'), **args)

            from matplotlib.ticker import FuncFormatter
            fmt = FuncFormatter(lambda x, pos: '{:.1%}'.format(x))
            args = {'vmin': 1e-7,
                    'vmax': 1.,
                    'cmap': 'cividis',
                    'title': "Relative uncertainty"}
            xu.plot_result(xu.get_rel_uncertainty(stat['mean'], stat['std']),
                           outname_base.format(key, 'rel_std'),
                           cbar_formatter=fmt, **args)

    if len(plottable_field_list) == 3:
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

        im = plt.imshow(xu.get_RGB_image_from_field(sky_field), origin="lower")
        rgb_filename = output_path + "sky_rgb.png"
        plt.savefig(rgb_filename, dpi=300)
        print(f"RGB image saved as {rgb_filename}.")
    else:
        print(f"RGB image not produced, {len(plottable_field_list)} colors were provided.")
