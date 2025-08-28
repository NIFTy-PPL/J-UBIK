import os
import sys
import pickle
from os.path import join

import numpy as np
import jax.numpy as jnp
from jax import linear_transpose, vmap, random
import matplotlib.pyplot as plt
from astropy import coordinates as coords
import astropy.io.fits as fits
import nifty.re as jft

import jubik0 as ju
from joss_paper_plotting import plot, plot_rgb


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

    bbox_info = [(7, 4), 28, 96,  'black']
    plot(real_pos,
         figsize=(7, 2.7),
         pixel_measure=112,
         fs=8,
         title=titles,
         logscale=True,
         colorbar=True,
         common_colorbar=True,
         n_rows=1,
         vmin=7e-7,
         vmax=5e-5,
         bbox_info=bbox_info,
         dpi=300,
         output_file=join(output_dir,
                          f'simulated_sky.png'),
         cbar_label=r"$\mathrm{s}^{-1}\mathrm{arcsec}^{-2}$",
         interpolation=None)

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

    # Chandra:
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

    # Full Plot
    plottabel_data_list = [
        unmasked_erosita_data[0],
        unmasked_chandra_data1[0],
        unmasked_chandra_data2[0],
    ]
    plottable_data = np.vstack(plottabel_data_list)
    title_list = ["eROSITA", "Chandra", "Chandra"]
    bbox_info = [(7, 4), 28, 96, "black"]
    pointing_center = [(512, 512), (512, 512), shifted_pointing_pix]
    plot(
        plottable_data,
        figsize=(7, 2.7),
        pixel_measure=112,
        fs=8,
        title=title_list,
        logscale=True,
        colorbar=True,
        common_colorbar=True,
        n_rows=1,
        vmin=5e1,
        vmax=5e3,
        dpi=300,
        bbox_info=bbox_info,
        pointing_center=pointing_center,
        output_file=join(output_dir, f"simulated_data.png"),
        cbar_label="counts",
    )

    # Zoom Plot
    plottabel_data_list = [unmasked_chandra_data1[0], unmasked_chandra_data2[0]]
    plottable_chandra_data = np.vstack(plottabel_data_list)
    title_list = ["Chandra", "Chandra"]
    pointing_center = [(512, 512), shifted_pointing_pix]
    plot(
        plottable_chandra_data,
        pixel_measure=112,
        fs=8,
        title=title_list,
        logscale=True,
        colorbar=True,
        common_colorbar=True,
        n_rows=1,
        vmin=5e1,
        vmax=5e3,
        dpi=300,
        bbox_info=bbox_info,
        pointing_center=pointing_center,
        output_file=join(output_dir, f"simulated_data_zoom.png"),
        cbar_label="counts",
    )
