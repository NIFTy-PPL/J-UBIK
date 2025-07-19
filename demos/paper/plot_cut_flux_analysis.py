import pickle
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

import jubik0 as ju
import nifty8.re as jft
from jax import config

config.update('jax_enable_x64', True)

# Define threshold
threshold_flux = 2.5e-9

# Load reconstruction config
results_path = './demos/results/LMC-03022025-001Mfull'
config_file = 'eROSITA_demo_full.yaml'
config = ju.get_config(join(results_path, config_file))

# Load response
response_dict = ju.build_erosita_response_from_config(config)

def response(x):
    return response_dict['exposure'](
        response_dict['psf'](x * response_dict['pix_area'],
        response_dict['kernel']))

# Load samples
with open(join(results_path, 'last.pkl'), 'rb') as f:
    samples, state = pickle.load(f)

# Create sky model
sky_model = ju.SkyModel(config)
sky = sky_model.create_sky_model()
sky_dict = sky_model.sky_model_to_dict()


# Get point source fields
point_sources = sky_dict['points']
diffuse = sky_dict['diffuse']
ps_mean, ps_std = jft.mean_and_std(tuple(point_sources(s) for s in samples))
sky_sr_mean = jft.mean(tuple(response(sky(s)) for s in samples))

ps_sr_mean = jft.mean(tuple(response(point_sources(s)) for s in samples))

# Create the mask based on ps_mean
mask = ps_mean > threshold_flux  # shape: (3, 1024, 1024)
masked_ps = np.ma.masked_where(mask, ps_mean)

masked_ps_sr = response(masked_ps.filled(0.))
summed_masked_ps_sr = np.sum(masked_ps_sr, axis=0)

ratio_with_noise = masked_ps_sr / np.sqrt(sky_sr_mean)


masked_ps_sr_tm1 = np.ma.masked_where(np.isnan(ratio_with_noise[0][0]), masked_ps_sr[0][0])


def plot_response_and_ratio(
    response_image,
    ratio_image,
    cmap_response="Blues_r",
    cmap_ratio="magma",
    response_title="response(cut point sources) TM1, 0.2 - 1.0 keV",
    ratio_title="ratio with noise, TM1, 0.2 - 1.0 keV",
    bbox_info= [(7, 4), 70,  120, 'black'],
    pixel_measure=112,
    pixel_factor=4,
    filename=None,
    dpi=300,
    left_colorbar_left=True,
    hide_ticks=True
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5),
                             sharex=True, sharey=True)

    # Optional tick removal
    if hide_ticks:
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

    # Left plot
    im0 = axes[0].imshow(response_image, origin='lower', cmap=cmap_response)
    # axes[0].set_title(response_title)

    if left_colorbar_left:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider0 = make_axes_locatable(axes[0])
        cax0 = divider0.append_axes("left", size="5%", pad=0.1)
        cbar0 = fig.colorbar(im0, cax=cax0, orientation='vertical')
        cbar0.ax.yaxis.set_ticks_position('left')
        cbar0.ax.yaxis.set_label_position('left')
    else:
        # Default right-side colorbar
        fig.colorbar(im0, ax=axes[0], orientation='vertical')

    # Right plot
    im1 = axes[1].imshow(ratio_image, origin='lower', cmap=cmap_ratio)
    # axes[1].set_title(ratio_title)
    if left_colorbar_left:
        divider1 = make_axes_locatable(axes[1])
        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        cbar1 = fig.colorbar(im1, cax=cax1, orientation='vertical')
        cbar1.ax.yaxis.set_ticks_position('right')
        cbar1.ax.yaxis.set_label_position('right')
    else:
        # Default right-side colorbar
        fig.colorbar(im1, ax=axes[1], orientation='vertical')

    if pixel_measure is not None:
        distance_measure = int(4 * pixel_measure / 60)
        x0, y0 = 10 * pixel_factor, 10 * pixel_factor
        x1, y1 = pixel_measure + 10 * pixel_factor, 10 * pixel_factor

        x2, y2 = 10 * pixel_factor, 8 * pixel_factor
        x3, y3 = 10 * pixel_factor, 12 * pixel_factor

        x4, y4 = pixel_measure + 10 * pixel_factor, 8 * pixel_factor
        x5, y5 = pixel_measure + 10 * pixel_factor, 12 * pixel_factor
        for i in range(2):
            rect = patches.Rectangle(bbox_info[0], pixel_measure + bbox_info[1],
                                     bbox_info[2], facecolor=bbox_info[3],
                                     alpha=0.5)
            axes[i].add_patch(rect)
            axes[i].text(int(pixel_measure / 2) + 7 * pixel_factor, 14 * pixel_factor,
                       f"{distance_measure}'", fontsize=12,
                       color='white')
            axes[i].plot([x0, x1], [y0, y1], color='white', linewidth=1)
            axes[i].plot([x2, x3], [y2, y3], color='white', linewidth=1)
            axes[i].plot([x4, x5], [y4, y5], color='white', linewidth=1)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=dpi)
        plt.close(fig)
        print(f"Saved plot to: {filename}")
    else:
        plt.show()


plot_response_and_ratio(
    masked_ps_sr_tm1,
    ratio_with_noise[0][0],
    filename="results/response_ratio_tm1.png"
)
