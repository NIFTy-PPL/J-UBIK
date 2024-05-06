import nifty8.re as jft

from typing import Dict, Callable

import numpy as np

from astropy.io import fits
from astropy.io.fits.hdu.image import ImageHDU
from astropy.time import Time

import time

from os.path import join
from os import makedirs


def _save_fits_2d(fld, filename):

    h = fits.Header()
    h["DATE-MAP"] = Time(time.time(), format="unix").iso.split()[0]

    # # FIXME: How to set the WCS, i.e. via distances ??
    # h["CRVAL1"] = h["CRVAL2"] = 0
    # h["CRPIX1"] = h["CRPIX2"] = 0
    # h["CUNIT1"] = h["CUNIT2"] = "deg"
    # h["CDELT1"], h["CDELT2"] = -distances[0], distances[1]
    # h["CTYPE1"] = "RA---SIN"
    # h["CTYPE2"] = "DEC---SIN"
    # h["EQUINOX"] = 2000

    hdu = fits.PrimaryHDU(fld, header=h)
    hdul = fits.HDUList([hdu])
    hdul.writeto(filename, overwrite=True)


def _export_operator(
    output_filename_base: str,
    operator: Callable,
    sample_list: jft.Samples,
    config: dict
) -> None:

    mean, std = jft.mean_and_std([operator(s) for s in sample_list])

    out_tmp = output_filename_base.format(suffix='')
    print(f'Saving results as fits files to: {out_tmp}')

    if config.get("mean", False):
        _save_fits_2d(
            np.log(mean) if config.get("log_scale", False) else mean,
            output_filename_base.format(suffix="mean")
        )

    if config.get("std", False):
        _save_fits_2d(
            std,
            output_filename_base.format(suffix="std")
        )

    if config.get("relative_std", False):
        _save_fits_2d(
            std / mean,
            output_filename_base.format(suffix="relative_std")
        )

    if config.get("samples", False):
        for index, s in enumerate(sample_list):
            _save_fits_2d(
                np.log(operator(s)) if config.get("log_scale", False) else
                operator(s),
                output_filename_base.format(suffix=f"sample_{index}")
            )


def export_operator_output_to_fits(
    output_directory: str,
    operators_dict: Dict[str, Callable],
    sample_list: jft.Samples,
    iteration: int = None,

    configs: dict = {},
) -> None:
    '''
    Exports the output of the operators to fits files.

    Parameters
    ----------
    - output_directory: `str`. The directory where the fits files will be saved.
    - operators_dict: `dict[callable]`. A dictionary containing operators.
    - sample_list: `nifty8.re.evi.Samples`. A list of samples.
    - iteration: `int`, optional. The global iteration number value. Defaults to None.

    - configs: `dict`, optional.
        - mean : `bool`, optional. Whether to save the mean of the samples. Defaults to True.
        - std: `bool`, optional. Whether to save the standard deviation. Defaults to False.
        - relative_std: `bool`, optional. Whether to save the relative standard deviation. Defaults to True.
        - samples: `bool`, optional. Whether to save the samples. Defaults to True.
        - log_scale: `bool`, optional. Whether to use a logarithmic scale. Defaults to False.
        - overwrite: `bool`, optional. Whether to overwrite existing files. Defaults to True.

    Returns
    -------
    - None
    '''

    subfolders = list(operators_dict.keys())
    for subfolder in subfolders:
        makedirs(join(output_directory, subfolder), exist_ok=True)

    for key, subfolder in zip(operators_dict, subfolders):
        config = configs.get(
            key,
            dict(mean=True, std=False, relative_std=True, samples=True,
                 log_scale=False, overwrite=True)
        )

        if config.get('overwrite', True):
            key_name = f'{key}_{{suffix}}.fits'
        else:
            key_name = f'{key}_{{suffix}}_{iteration}.fits'

        _export_operator(
            join(output_directory, subfolder, key_name),
            operators_dict[key],
            sample_list,
            config
        )
