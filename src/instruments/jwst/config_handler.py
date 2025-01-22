# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2024 Max-Planck-Society

# %
from .grid import Grid
from ...hashcollector import save_local_packages_hashes_to_txt
from ...utils import save_config_copy_easy

import os
import yaml

from astropy import units as u


def load_yaml_and_save_info(config_path):
    cfg = yaml.load(open(config_path, 'r'), Loader=yaml.SafeLoader)
    results_directory = cfg['files']['res_dir']
    os.makedirs(results_directory, exist_ok=True)
    save_local_packages_hashes_to_txt(
        ['nifty8', 'charm_lensing', 'jubik0'],
        os.path.join(results_directory, 'hashes.txt'))
    save_config_copy_easy(config_path, os.path.join(
        results_directory, 'config.yaml'))
    return cfg, results_directory


def config_transform(config: dict):
    """
    Recursively transforms string values in a configuration dictionary.

    This function processes a dictionary and attempts to evaluate any string
    values that may represent valid Python expressions. If the string cannot
    be evaluated, it is left unchanged. The function also applies the same
    transformation recursively for any nested dictionaries.

    Parameters
    ----------
    config : dict
        The configuration dictionary where string values may be transformed.
        If a value is a string that can be evaluated, it is replaced by the
        result of `eval(val)`. Nested dictionaries are processed recursively.
    """
    for key, val in config.items():
        if isinstance(val, str):
            try:
                config[key] = eval(val)
            except:
                continue
        elif isinstance(val, dict):
            config_transform(val)


def get_grid_extension_from_config(
    telescope_config: dict,
    reconstruction_grid: Grid,
):
    '''Load the grid extension for the reconstruction grid. The reconstruction
    gets zero padded by this amount. This is needed to avoid wrapping flux due
    to fft convolution of the psf.

    Parameters
    ----------
    telescope_config: dict
        The telescope_config dict holds psf_arcsec_extension, which holds the
        extension in units of arcsec.
    reconstruction_grid: Grid
        The grid underlying the reconstruction.

    Returns
    -------
    grid_extension: tuple[int]
        A pixel number tuple, that specifies by how many pixels the
        reconstruction will be zero padded.
    '''

    psf_arcsec_extension = telescope_config['psf'].get('psf_arcsec_extension')
    if psf_arcsec_extension is None:
        raise ValueError('Need to provide either `psf_arcsec_extension`.')

    return [int((psf_arcsec_extension*u.arcsec).to(u.deg) / 2 / dist)
            for dist in reconstruction_grid.spatial.distances]


def _parse_insert_spaces(cfg):
    lens_fov = cfg['grid']['fov']
    lens_npix = cfg['grid']['sdim']
    lens_padd = cfg['grid']['s_padding_ratio']
    lens_npix = (lens_npix, lens_npix)
    lens_dist = [lens_fov/p for p in lens_npix]
    lens_energy_bin = cfg['grid']['energy_bin']
    lens_space = dict(padding_ratio=lens_padd,
                      Npix=lens_npix,
                      distance=lens_dist,
                      energy_bin=lens_energy_bin
                      )

    source_fov = cfg['grid']['source_grid']['fov']
    source_npix = cfg['grid']['source_grid']['sdim']
    source_padd = cfg['grid']['source_grid']['s_padding_ratio']
    source_npix = (source_npix, source_npix)
    source_dist = [source_fov/p for p in source_npix]
    source_energy_bin = cfg['grid']['energy_bin']
    source_space = dict(padding_ratio=source_padd,
                        Npix=source_npix,
                        distance=source_dist,
                        energy_bin=source_energy_bin,
                        )

    return lens_space, source_space


def insert_spaces_in_lensing(cfg):
    lens_space, source_space = _parse_insert_spaces(cfg)

    cfg['lensing']['spaces'] = dict(
        lens_space=lens_space, source_space=source_space)


def insert_spaces_in_lensing_new(cfg):
    lens_space, source_space = _parse_insert_spaces(cfg)

    cfg['spaces'] = dict(
        lens_space=lens_space, source_space=source_space)
