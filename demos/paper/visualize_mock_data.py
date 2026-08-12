"""
TODO
- [ ] relocate to paper? or where it makes more sense
- [ ] check .T in dorados and paper plotting for future datafusion
- [ ] cleanup
- [ ] to main branch (quasi)
"""

import os
import sys
import pickle
from os.path import join

import numpy as np
import jax.numpy as jnp
from jax import linear_transpose, vmap, random

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

from astropy import coordinates as coords
import astropy.io.fits as fits
import nifty.re as jft

import jubik0 as ju
from joss_paper_plotting import joss_figsize, joss_scalebar, joss_cmap, joss_colorbar

mplstyle.use("paper/joss.mplstyle")

# --- figsize Helper ---

def fourier_conv_no_size(a, b):
    """Fourier convolution with pixel size"""
    a_k = jnp.fft.fftn(a)
    b_k = jnp.fft.fftn(b)
    c_k = a_k*b_k
    c = jnp.fft.ifftn(c_k)
    return jnp.abs(c)

def gaussian_psf_full(shape=(1024, 1024), sigma=2.0):
    """
    Generates an isotropic Gaussian-PSFon a full grid and a given standard deviation.
    The PSF is correctly centered for a fft based convolution.

    Parameters
    ----------
    shape: tuple(int, int)
        describing the shape of the full grid.
    sigma: float
        standard deviation of the Gaussian PSF

    Return
    ------
    Gaussian PSF, not normalized
    """
    H, W = shape

    y = np.arange(H) - H // 2
    x = np.arange(W) - W // 2
    Y, X = np.meshgrid(y, x, indexing="ij")

    psf = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    psf /= psf.sum()

    psf = np.fft.ifftshift(psf)
    return psf


def to_slice_coords(x, y, x_start, y_start):
    if x < x_start or y < y_start:
        return None
    return x - x_start, y - y_start


# Script for plotting the data, position and reconstruction images
if __name__ == "__main__":
    output_dir = ju.create_output_directory("paper/")
    key = random.PRNGKey(81)

    # Read config files
    prior_config_path = "paper/prior_config.yaml"
    prior_config_dict = ju.get_config(prior_config_path)

    eROSITA_config_name = "paper/eROSITA_demo.yaml"
    eROSITA_cfg_dict = ju.get_config(eROSITA_config_name)

    chandra_config_name1 = "paper/chandra_demo_1.yaml"
    chandra_cfg_dict1 = ju.get_config(chandra_config_name1)

    chandra_config_name2 = "paper/chandra_demo_2.yaml"
    chandra_cfg_dict2 = ju.get_config(chandra_config_name2)

    # build sky model
    sky_model = ju.SkyModel(prior_config_dict)
    sky = sky_model.create_sky_model()
    sky_dict = sky_model.sky_model_to_dict()

    # random latent position for sky
    key, subkey = random.split(key)
    pos = jft.Vector(jft.random_like(subkey, sky.domain))

    factor = 100
    real_pos = []
    titles = []
    for dict_key, op in sky_dict.items():
        real_pos.append(factor*op(pos))
        titles.append(dict_key)
    real_pos = np.vstack(real_pos)
    np.testing.assert_allclose(real_pos[0], factor*sky(pos)[0])


    # Blur points and add to diffuse for and plot these two for better printing
    psf = gaussian_psf_full((1024,1024), 1)
    psf = psf/psf.sum()
    blurred_ps = fourier_conv_no_size(real_pos[2], psf)
    real_pos[2] = blurred_ps
    real_pos[0] = real_pos[1] + real_pos[2]
    # Plotting
    fig, ax = plt.subplots(1,3,figsize=joss_figsize(aspect_ratio=0.5), constrained_layout=True)
    for i in range(3):
        im = ax[i].imshow(real_pos[i], norm="log",vmin=1e-7, vmax=1e-5, cmap=joss_cmap())
        joss_scalebar(ax[i], 8*60/4,"8 arcmin", "lower right")
    joss_colorbar(fig, im, ax=ax, fraction=0.5, pad=0.03, label=r"$\mathrm{s}^{-1}\mathrm{arcsec}^{-2}$", orientation="horizontal", labelpad=3)
    fig.savefig("paper/simulated_sky.pdf")


    fig = plt.figure(figsize=(6.5, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 0.1])

    ax_top = fig.add_subplot(gs[0, :])
    ax1    = fig.add_subplot(gs[1, 0])
    ax2    = fig.add_subplot(gs[1, 1])
    cax    = fig.add_subplot(gs[2, :])

    # quadratisch halten
    for ax in [ax_top, ax1, ax2]:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        joss_scalebar(ax, 8*60/4,"8 arcmin", "lower right")

    im=ax_top.imshow(real_pos[0], norm="log",vmin=1e-7, vmax=1e-5, cmap=joss_cmap())
    ax1.imshow(real_pos[1], norm="log",vmin=1e-7, vmax=1e-5, cmap=joss_cmap())
    ax2.imshow(real_pos[2], norm="log",vmin=1e-7, vmax=1e-5, cmap=joss_cmap())
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label(r"$\mathrm{s}^{-1}\mathrm{arcsec}^{-2}$")
    # aa_colorbar(fig, im, ax=gs, fraction=0.5, pad=0.03, label=r"$\mathrm{s}^{-1}\mathrm{cm}^{2}$", orientation="horizontal", labelpad=3)

    # fig.savefig("paper/simulated_sky_test.pdf") # uncomment for big version

    #bbox_info = [(7, 4), 28, 96,  'black']
    # plot(real_pos,
    #      figsize=(7, 2.7),
    #      pixel_measure=112,
    #      fs=8,
    #      title=titles,
    #      logscale=True,
    #      colorbar=True,
    #      common_colorbar=True,
    #      n_rows=1,
    #      vmin=7e-7,
    #      vmax=5e-5,
    #      bbox_info=bbox_info,
    #      dpi=300,
    #      output_file=join(output_dir,
    #                       f'simulated_sky.png'),
    #      cbar_label=r"$\mathrm{s}^{-1}\mathrm{arcsec}^{-2}$",
    #      interpolation=None)
    # eROSITA:
    response_dict = ju.build_erosita_response_from_config(eROSITA_cfg_dict)
    masked_mock_data = response_dict["R"](factor * sky(pos), response_dict["kernel"])

    # Poisson counts
    poisson1, poisson2, poisson3 = random.split(key, 3)
    masked_mock_data = jft.Vector(
        {
            tm: random.poisson(poisson1, data).astype(int)
            for i, (tm, data) in enumerate(masked_mock_data.tree.items())
        }
    )
    plottable_vector = jft.Vector(
        {key: val.astype(float) for key, val in masked_mock_data.tree.items()}
    )
    mask = response_dict["mask"]
    mask_adj = linear_transpose(mask, np.zeros((1, 1, 1024, 1024)))
    mask_adj_func = lambda x: mask_adj(x)[0]

    # Plotting the data
    unmasked_erosita_data = mask_adj_func(plottable_vector)

    probe_masks = np.ones((1, 1, 1024, 1024))
    erosita_mask_contour = mask_adj_func(mask(probe_masks))[0,0]

    # Chandra1:
    response_dict = ju.build_chandra_response_from_config(chandra_cfg_dict1)
    masked_mock_data = response_dict["R"](factor * sky(pos))

    masked_mock_data = jft.Vector(
        {
            tm: random.poisson(poisson2, data).astype(int)
            for i, (tm, data) in enumerate(masked_mock_data.tree.items())
        }
    )

    plottable_vector = jft.Vector(
        {key: val.astype(float) for key, val in masked_mock_data.tree.items()}
    )

    mask = response_dict["mask"]
    mask_adj = linear_transpose(mask, np.zeros((1, 1, 1024, 1024)))
    mask_adj_func = lambda x: mask_adj(x)[0]

    # Plotting the data
    unmasked_chandra_data1 = mask_adj_func(plottable_vector)

    chandra1_mask_contour = mask_adj_func(mask(probe_masks))[0,0]

    # Chandra2:
    response_dict = ju.build_chandra_response_from_config(chandra_cfg_dict2)
    masked_mock_data = response_dict["R"](factor * sky(pos))

    masked_mock_data = jft.Vector(
        {
            tm: random.poisson(poisson3, data).astype(int)
            for i, (tm, data) in enumerate(masked_mock_data.tree.items())
        }
    )
    plottable_vector = jft.Vector(
        {key: val.astype(float) for key, val in masked_mock_data.tree.items()}
    )
    mask = response_dict["mask"]
    mask_adj = linear_transpose(mask, np.zeros((1, 1, 1024, 1024)))
    mask_adj_func = lambda x: mask_adj(x)[0]

    # Plotting the data
    unmasked_chandra_data2 = mask_adj_func(plottable_vector)
    chandra2_mask_contour = mask_adj_func(mask(probe_masks))[0,0]


    # TODO Document center correction
    center = np.array((49.9412, 41.5278))
    shifted_pointing = np.array((49.8770, 41.6287))

    pointing_stats = coords.SkyCoord(
        ra=shifted_pointing[0], dec=shifted_pointing[1], unit="deg", frame="icrs"
    )
    # center with respect to desired pointing center
    ref_center = coords.SkyCoord(
        ra=center[0], dec=center[1], unit="deg", frame="icrs"
    )  # TODO Check Frame
    d_centers_astropy = pointing_stats.transform_to(
        coords.SkyOffsetFrame(origin=ref_center)
    )
    d_centers = np.array([d_centers_astropy.lon.arcsec, d_centers_astropy.lat.arcsec])
    d_pix = d_centers / 4
    shifted_pointing_pix = (512 + d_pix[1], 512 - d_pix[0])
    pointing_center = [(512, 512), (512, 512), shifted_pointing_pix]

    ch1_x_start, ch1_x_end, ch1_y_start, ch1_y_end = 250, 800, 350, 900
    ch2_x_start, ch2_x_end, ch2_y_start, ch2_y_end = 250, 800, 250, 800

    p_ch1 = to_slice_coords(512,512,ch1_x_start, ch1_y_start)
    p_ch2 = to_slice_coords(shifted_pointing_pix[1],shifted_pointing_pix[0],ch2_x_start, ch2_y_start)

    # Full Plot
    plottabel_data_list = [
        unmasked_erosita_data[0],
        unmasked_chandra_data1[0],
        unmasked_chandra_data2[0],
    ]
    plottable_data = np.vstack(plottabel_data_list)

    fig, ax = plt.subplots(2,2,figsize=joss_figsize(aspect_ratio=1.1), constrained_layout=True)
    im = ax[0,0].imshow(plottable_data[0], norm="log",vmin=1, vmax=1e3, cmap=joss_cmap())    
    ax[0,0].plot(512,512,
                marker="+",
                color="red",
                markersize=5,
                markeredgewidth=0.5,
            )
    ax[0,1].imshow(plottable_data[1, ch1_x_start:ch1_x_end, ch1_y_start:ch1_y_end], norm="log",vmin=1, vmax=1e3, cmap=joss_cmap())
    ax[0,1].plot(p_ch1[1],p_ch1[0],
                marker="+",
                color="red",
                markersize=5,
                markeredgewidth=0.5,
            )
    ax[1,0].imshow(plottable_data[2, ch2_x_start:ch2_x_end, ch2_y_start:ch2_y_end], norm="log",vmin=1, vmax=1e3, cmap=joss_cmap())
    ax[1,0].plot(p_ch2[1],p_ch2[0],
                marker="+",
                color="red",
                markersize=5,
                markeredgewidth=0.5,
            )
    ax[1,1].imshow(real_pos[0]*1e8, norm="log",vmin=1, vmax=1e3, cmap=joss_cmap())
    ax[1,1].contour(erosita_mask_contour, linewidths=.2, colors="orange")
    ax[1,1].contour(chandra1_mask_contour,linewidths=.2, colors="blue")
    ax[1,1].contour(chandra2_mask_contour,linewidths=.2, colors="white")

    for i in range(4):
        ax = ax.flatten()
        joss_scalebar(ax[i], 8*60/4,"8 arcmin", "lower right")
    joss_colorbar(fig, im, ax=ax, fraction=0.5, pad=0.03, label="counts", orientation="horizontal", labelpad=3)
    fig.savefig("paper/simulated_data.pdf")



    # title_list = ["eROSITA", "Chandra", "Chandra"]
    # bbox_info = [(7, 4), 28, 96, "black"]
    # plot(
    #     plottable_data,
    #     figsize=(7, 2.7),
    #     pixel_measure=112,
    #     fs=8,
    #     title=title_list,
    #     logscale=True,
    #     colorbar=True,
    #     common_colorbar=True,
    #     n_rows=1,
    #     vmin=5e1,
    #     vmax=5e3,
    #     dpi=300,
    #     bbox_info=bbox_info,
    #     pointing_center=pointing_center,
    #     output_file=join(output_dir, f"simulated_data.png"),
    #     cbar_label="counts",
    # )

    # Zoom Plot
    # plottabel_data_list = [unmasked_chandra_data1[0], unmasked_chandra_data2[0]]
    # plottable_chandra_data = np.vstack(plottabel_data_list)
    # title_list = ["Chandra", "Chandra"]
    # pointing_center = [(512, 512), shifted_pointing_pix]
    # plot(
    #     plottable_chandra_data,
    #     pixel_measure=112,
    #     fs=8,
    #     title=title_list,
    #     logscale=True,
    #     colorbar=True,
    #     common_colorbar=True,
    #     n_rows=1,
    #     vmin=5e1,
    #     vmax=5e3,
    #     dpi=300,
    #     bbox_info=bbox_info,
    #     pointing_center=pointing_center,
    #     output_file=join(output_dir, f"simulated_data_zoom.png"),
    #     cbar_label="counts",
    # )
