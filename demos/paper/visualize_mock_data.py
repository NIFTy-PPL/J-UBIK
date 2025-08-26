import matplotlib.pyplot as plt
import numpy as np
import pickle
from jax import linear_transpose, vmap, random
import jax.numpy as jnp
from astropy import coordinates as coords
import astropy.io.fits as fits

import jubik0 as ju
from os.path import join
import nifty.re as jft
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from joss_paper_plotting import plot, plot_rgb


# Script for plotting the data, position and reconstruction images
if __name__ == "__main__":
    output_dir = ju.create_output_directory("paper/")

    eROSITA_config_name = "paper/eROSITA_demo.yaml"
    chandra_config_name1 = "paper/chandra_demo_1.yaml"
    chandra_config_name2 = "paper/chandra_demo_2.yaml"
    prior_config_path = "paper/prior_config.yaml"

    eROSITA_cfg_dict = ju.get_config(eROSITA_config_name)
    chandra_cfg_dict1 = ju.get_config(chandra_config_name1)
    chandra_cfg_dict2 = ju.get_config(chandra_config_name2)
    prior_config_dict = ju.get_config(prior_config_path)

    sky_model = ju.SkyModel(prior_config_dict)
    sky = sky_model.create_sky_model()

    pos = ju.load_from_pickle('paper/pos.pkl')
    factor = 100

    # eROSITA:
    response_dict = ju.build_erosita_response_from_config(eROSITA_cfg_dict)
    masked_mock_data = response_dict['R'](factor*sky(pos), response_dict['kernel'])
    key = random.PRNGKey(67)
    key, subkey = random.split(key)
    masked_mock_data = jft.Vector({
        tm: random.poisson(subkey, data).astype(int)
        for i, (tm, data) in enumerate(masked_mock_data.tree.items())
    })
    plottable_vector = jft.Vector({key: val.astype(float) for key, val
                                   in masked_mock_data.tree.items()})
    mask = response_dict['mask']
    mask_adj = linear_transpose(mask,
                                np.zeros((1, 1, 1024, 1024)))
    mask_adj_func = lambda x: mask_adj(x)[0]
    tms = plottable_vector.tree.keys()
    # Plotting the data
    unmasked_erosita_data = mask_adj_func(plottable_vector)

    # Chandra:
    response_dict = ju.build_chandra_response_from_config(chandra_cfg_dict1)
    masked_mock_data = response_dict['R'](factor*sky(pos))
    key = random.PRNGKey(67)
    key, subkey = random.split(key)
    masked_mock_data = jft.Vector({
        tm: random.poisson(subkey, data).astype(int)
        for i, (tm, data) in enumerate(masked_mock_data.tree.items())
    })
    plottable_vector = jft.Vector({key: val.astype(float) for key, val
                                   in masked_mock_data.tree.items()})
    mask = response_dict['mask']
    mask_adj = linear_transpose(mask,
                                np.zeros((1, 1, 1024, 1024)))
    mask_adj_func = lambda x: mask_adj(x)[0]
    tms = plottable_vector.tree.keys()
    # Plotting the data
    unmasked_chandra_data1 = mask_adj_func(plottable_vector)

    response_dict = ju.build_chandra_response_from_config(chandra_cfg_dict2)
    masked_mock_data = response_dict['R'](factor*sky(pos))
    key = random.PRNGKey(67)
    key, subkey = random.split(key)
    masked_mock_data = jft.Vector({
        tm: random.poisson(subkey, data).astype(int)
        for i, (tm, data) in enumerate(masked_mock_data.tree.items())
    })
    plottable_vector = jft.Vector({key: val.astype(float) for key, val
                                   in masked_mock_data.tree.items()})
    mask = response_dict['mask']
    mask_adj = linear_transpose(mask,
                                np.zeros((1, 1, 1024, 1024)))
    mask_adj_func = lambda x: mask_adj(x)[0]
    tms = plottable_vector.tree.keys()
    # Plotting the data
    unmasked_chandra_data2 = mask_adj_func(plottable_vector)

    from astropy import coordinates as coords
    center = np.array((49.9412, 41.5278))
    shifted_pointing = np.array((49.8770, 41.6287))

    pointing_stats = coords.SkyCoord(ra=shifted_pointing[0],
                                   dec=shifted_pointing[1],
                                   unit='deg',
                                   frame='icrs')
    # center with respect to desired pointing center
    ref_center = coords.SkyCoord(ra=center[0],
                                 dec=center[1],
                                 unit='deg',
                                 frame='icrs') # TODO Check Frame
    d_centers_astropy = pointing_stats.transform_to(
        coords.SkyOffsetFrame(origin=ref_center))
    d_centers = np.array(
        [d_centers_astropy.lon.arcsec, d_centers_astropy.lat.arcsec]
    )
    d_pix = d_centers/4
    shifted_pointing_pix = (512+d_pix[1], 512-d_pix[0])
    print(d_pix)

    plottabel_data_list = [unmasked_erosita_data[0], unmasked_chandra_data1[0],
                           unmasked_chandra_data2[0]]
    plottable_data = np.vstack(plottabel_data_list)
    title_list = ['eROSITA', 'Chandra', 'Chandra']
    bbox_info = [(7, 4), 28, 96,  'black']
    pointing_center = [(512, 512), (512, 512), shifted_pointing_pix]
    plot(plottable_data,
         pixel_measure=112,
         fs=8,
         title=title_list,
         logscale=True,
         colorbar=True,
         common_colorbar=True,
         n_rows=1,
         vmin=5e1,
         vmax=5e3,
         dpi=1500,
         bbox_info=bbox_info,
         pointing_center = pointing_center,
         output_file=join(output_dir,
         f'simulated_data.png'))

    plottabel_data_list = [unmasked_chandra_data1[0],
                           unmasked_chandra_data2[0]]
    plottable_chandra_data = np.vstack(plottabel_data_list)
    title_list = ['Chandra', 'Chandra']
    pointing_center = [(512, 512), shifted_pointing_pix]
    plot(plottable_chandra_data,
         pixel_measure=112,
         fs=8,
         title=title_list,
         logscale=True,
         colorbar=True,
         common_colorbar=True,
         n_rows=1,
         vmin=5e1,
         vmax=5e3,
         dpi=1500,
         bbox_info=bbox_info,
         pointing_center = pointing_center,
         output_file=join(output_dir,
         f'simulated_data_zoom.png'))

