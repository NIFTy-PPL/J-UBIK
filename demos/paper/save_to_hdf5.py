# Save to HDF5
import sys
import os

import h5py
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cbook
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, \
    zoomed_inset_axes
import pickle
from jax import linear_transpose, vmap
import jax.numpy as jnp
from jax import config
import jax
from pathlib import Path

import jubik0 as ju
from os.path import join, basename
import astropy.io.fits as fits

import nifty8.re as jft

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_eROSITA_image import plot, plot_rgb


def save_arrays_to_hdf5_with_groups(file_path, grouped_data):
    """
    Speichert strukturierte NumPy-Arrays in HDF5-Datei mit Gruppen und Beschreibungen.

    Parameters:
    - file_path (str): Pfad zur HDF5-Datei
    - grouped_data (dict): Dictionary mit Gruppenname als Key und weiteren Dictionaries als Value:
        {
            "gruppe1": {
                "array1": {"data": ..., "description": "..."},
                "array2": {"data": ..., "description": "..."}
            },
            "gruppe2": {
                ...
            }
        }
    """
    with h5py.File(file_path, 'w') as h5file:
        for group_name, arrays in grouped_data.items():
            group = h5file.create_group(group_name)

            for array_name, content in arrays.items():
                data = content['data']
                description = content.get('description', '')

                dset = group.create_dataset(array_name, data=data)
                dset.attrs['description'] = description

    print(f"Daten erfolgreich in '{file_path}' gespeichert.")


config.update('jax_enable_x64', True)

# Script for plotting the data, position and reconstruction images
results_path = "results/LMC_rec_22092025_M6"
config_name = "eROSITA_demo6.yaml"
path_to_caldb = '../data/'
output_dir = ju.create_output_directory(join(results_path, 'paper'))
config_path = join(results_path, config_name)
config_dict = ju.get_config(config_path)
tel_info = config_dict["telescope"]
tm_ids = tel_info["tm_ids"]

grid_info = config_dict["grid"]
epix = grid_info['edim']
spix = grid_info['sdim']
e_min = grid_info['energy_bin']['e_min']
e_max = grid_info['energy_bin']['e_max']

with open(os.path.join(results_path, 'last.pkl'), "rb") as file:
    samples, _ = pickle.load(file)

sky_model = ju.SkyModel(config_dict)
sky = sky_model.create_sky_model()
sky_dict = sky_model.sky_model_to_dict()

file_info = config_dict['files']
exposure_filenames = []
for tm_id in tel_info['tm_ids']:
    exposure_filename = f'tm{tm_id}_' + file_info['exposure']
    [exposure_filenames.append(join(file_info['obs_path'],
                                    "processed",
                                    f"{Path(exposure_filename).stem}_emin{e}_emax{E}.fits"))
     for e, E in zip(e_min, e_max)]

exposures = []
tm_id = basename(exposure_filenames[0])[2]
tm_exposures = []
for file in exposure_filenames:
    if basename(file)[2] != tm_id:
        exposures.append(tm_exposures)
        tm_exposures = []
        tm_id = basename(file)[2]
    if basename(file)[2] == tm_id:
        if file.endswith('.npy'):
            tm_exposures.append(np.load(file))
        elif file.endswith('.fits'):
            tm_exposures.append(fits.open(file)[0].data)
        elif not (file.endswith('.npy') or file.endswith('.fits')):
            raise ValueError('exposure files should be in a .npy or .fits format!')
        else:
            raise FileNotFoundError(f'cannot find {file}!')
exposures.append(tm_exposures)
exposures = np.array(exposures, dtype="float64")
summed_exposure = np.sum(exposures, axis=0)

def mask(x):
    masked_x = x.at[summed_exposure<=500].set(0)
    return masked_x

dct = {}
for key, op in sky_dict.items():
    if key == "masked_diffuse":
        key = "doradus_c"
    op = jax.vmap(op)
    real_samples = op(samples.samples)
    real_mean = jnp.mean(real_samples, axis=0)
    real_std = jnp.std(real_samples, axis=0)
    temp_dict = {"mean":
                    {"data": real_mean,
                     "description": f"{key} mean, shape: energy, x, y"},
                 "std":
                    {"data": real_std,
                     "description": f"{key} standard deviation, shape: energy, x, y"},
                 "samples":
                    {"data": real_samples,
                     "description": f"{key} samples, shape: samples, energy, x, y"},
                 }
    dct.update({key: temp_dict})

save_arrays_to_hdf5_with_groups(output_dir+"/TM6.h5", dct)
