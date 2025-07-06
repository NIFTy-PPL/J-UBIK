# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2024 Max-Planck-Society

# %
from ...grid import Grid
from ...parse.wcs.spatial_model import yaml_dict_to_fov, yaml_dict_to_shape
from ...hashcollector import save_local_packages_hashes_to_txt
from ...utils import save_config_copy_easy

import os
import yaml

from astropy import units as u


def load_yaml_and_save_info(config_path):
    cfg = yaml.load(open(config_path, "r"), Loader=yaml.SafeLoader)

    results_directory = cfg["files"]["res_dir"]
    if cfg["files"].get("set_filter", False):
        filter_mapping = dict(
            f2100w="01",
            f1800w="02",
            f1500w="03",
            f1280w="04",
            f1000w="05",
            f770w="06",
            f560w="07",
            f444w="08",
            f356w="09",
            f277w="10",
            f200w="11",
            f150w="12",
            f115w="13",
        )
        filter_name = next(iter(cfg["files"]["filter"]))
        filter = f"{filter_mapping[filter_name]}_{filter_name}"
        results_directory = results_directory.format(filter=filter)

    os.makedirs(results_directory, exist_ok=True)

    save_local_packages_hashes_to_txt(
        ["nifty8", "charm_lensing", "jubik0"],
        os.path.join(results_directory, "hashes.txt"),
    )
    save_config_copy_easy(config_path, os.path.join(results_directory, "config.yaml"))
    return cfg, results_directory


def get_grid_extension_from_config(
    telescope_config: dict,
    reconstruction_grid: Grid,
) -> tuple[int, int]:
    """Load a pixelwise extension of the reconstruction grid. The reconstruction grid
    will be extended by half the grid extension in both spatial dimensions.

    This extension is needed in order to avoid flux being wrapped around the periodic
    boundary of the grid by the fft-psf convolution.

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
    """

    psf_arcsec_extension = telescope_config["psf"].get("psf_arcsec_extension")
    if psf_arcsec_extension is None:
        raise ValueError("Need to provide either `psf_arcsec_extension`.")

    return [
        int((psf_arcsec_extension * u.arcsec).to(u.deg) / 2 / dist)
        for dist in reconstruction_grid.spatial.distances
    ]


def _parse_insert_spaces(cfg):
    lens_fov = yaml_dict_to_fov(cfg["grid"])
    lens_npix = yaml_dict_to_shape(cfg["grid"])
    lens_padd = cfg["grid"]["s_padding_ratio"]
    lens_distances = [
        fov.to(u.arcsec).value / pix for fov, pix in zip(lens_fov, lens_npix)
    ]
    lens_energy_bin = cfg["grid"]["energy_bin"]
    lens_space = dict(
        padding_ratio=lens_padd,
        Npix=lens_npix,
        distance=lens_distances,
        energy_bin=lens_energy_bin,
    )

    source_fov = yaml_dict_to_fov(cfg["grid"]["source_grid"])
    source_npix = yaml_dict_to_shape(cfg["grid"]["source_grid"])
    source_padd = cfg["grid"]["source_grid"]["s_padding_ratio"]
    source_distances = [
        fov.to(u.arcsec).value / pix for fov, pix in zip(source_fov, source_npix)
    ]
    source_energy_bin = cfg["grid"]["energy_bin"]
    source_space = dict(
        padding_ratio=source_padd,
        Npix=source_npix,
        distance=source_distances,
        energy_bin=source_energy_bin,
    )

    interpolation = cfg["grid"]["source_grid"].get("interpolation", "bilinear")

    return lens_space, source_space, interpolation


def insert_spaces_in_lensing_new(cfg):
    lens_space, source_space, interpolation = _parse_insert_spaces(cfg)

    cfg["spaces"] = dict(
        lens_space=lens_space, source_space=source_space, interpolation=interpolation
    )
