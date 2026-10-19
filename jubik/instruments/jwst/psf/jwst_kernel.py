# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2024 Max-Planck-Society

# %
from os import makedirs
from os.path import isfile, join

import numpy as np
from astropy.coordinates import SkyCoord
from nifty.re import logger
from numpy.typing import ArrayLike

from ..data.jwst_data import JwstData
from ..parse.jwst_psf import JwstPsfKernelConfig


# --------------------------------
# Testing seam
# --------------------------------
# `test/instruments/jwst/test_jwst_likelihoods.py` monkeypatches `build_webb_psf` so the
# likelihood smoke test never imports stpsf. Keep the stpsf call confined to this
# function so that one patch is enough.
def build_webb_psf(
    camera: str,
    filter: str,
    center_pixel: tuple[float],
    webbpsf_path: str,
    subsample: int,
    fov_arcsec: float | None = None,
    normalize: str = "last",
):
    """
    Builds a Point Spread Function (PSF) model for the JWST using the specified
    camera and filter.

    This function computes the PSF using the `stpsf` library.
    It requires specifying the camera, filter, and center pixel position.
    You must also provide either the field of view (FOV) in pixels or
    in arcseconds.
    The resulting PSF can be used for further analysis or simulations.

    Parameters
    ----------
    camera : str
        The camera for which to compute the PSF. Supported options are 'nircam'
        and 'miri'.
    filter : str
        The filter for which to compute the PSF.
    center_pixel : tuple of float
        The position of the center pixel (x, y) for the PSF calculation.
    webbpsf_path : str
        The directory holding the STPSF calibration data. It is passed to
        STPSF via the `STPSF_PATH` environment variable.
    subsample : int
        The oversampling factor for the PSF computation.
        This determines how finely the PSF is sampled.
    fov_arcsec : float | None
        The field of view (FOV) in arcseconds.
        If provided, it specifies the size of the PSF image
        in arcseconds.
    normalize : str | None
        The normalization method for the PSF.
        Default is 'last', which normalizes based on the
        last computed value. Other options may be supported by `stpsf`.

    Returns
    -------
    numpy.ndarray
        The computed PSF data as a 2D array.

    Raises
    ------
    ImportError
        If `stpsf` is not installed.
    KeyError
        If the specified `camera` is not supported.

    Notes
    -----
    - Ensure that the `stpsf` library is properly installed and that
      `webbpsf_path` points to the STPSF calibration data directory.
    - The PSF data is returned as a 2D numpy array.

    Example
    -------
    To build a PSF for the NIRCam camera using the F090W filter,
    centered at pixel (100, 100), with a field of view of 256 pixels and an
    oversampling factor of 3:

    >>> psf = build_webb_psf(
    >>>     camera='nircam',
    >>>     filter='F090W',
    >>>     center_pixel=(100, 100),
    >>>     webbpsf_path='/path/to/webbpsf/data',
    >>>     subsample=3,
    >>>     fov_pixels=256
    >>> )
    """
    try:
        import stpsf
    except ImportError as ie:
        raise ImportError("stpsf is not installed. Please install it first") from ie
    from os import environ

    # STPSF reads its calibration directory from `STPSF_PATH` when an
    # instrument is initialized, so it has to be set before that.
    environ["STPSF_PATH"] = webbpsf_path

    psf_model_supported = {
        "nircam": stpsf.NIRCam,
        "miri": stpsf.MIRI,
    }

    try:
        psf_model_class = psf_model_supported[camera.lower()]
    except KeyError as ke:
        raise KeyError(
            f"You requested the {camera.lower()} STPSF model. "
            f"Supported STPSF models: {list(psf_model_supported)}"
        ) from ke
    psf_model = psf_model_class()

    psf_model.filter = filter.upper()
    psf_model.detector_position = center_pixel

    psf = psf_model.calc_psf(
        fov_pixels=None,
        fov_arcsec=fov_arcsec,
        oversample=subsample,
        normalize=normalize,
    )

    return psf[2].data


def load_psf_kernel(
    jwst_data: JwstData,
    subsample: int,
    target_center: SkyCoord,
    config_parameters: JwstPsfKernelConfig,
) -> ArrayLike:
    """
    Loads or computes the Point Spread Function (PSF) kernel for a specified
    camera and filter.

    This function attempts to load a precomputed PSF kernel from a specified
    library path.
    If the PSF kernel file does not exist, it computes the PSF using the
    `build_webb_psf` function, saves the result to the specified library path,
    and then returns the PSF data.

    Parameters
    ----------
    jwst_data: JwstData
        jwst data with camera and filter
    target_center: SkyCoord
        The center of the observation for which the psf will be evaluted.
        The psf kernel is assumed to be static across the field.
    subsample: int
        The subsample factor for the psf kernel.
    config_parameters: PsfKernelConfig
        Holding the `webbpsf_path`, the `psf_library_path`, the size of the psf
        evaluation `psf_arcsec`, and the `normalize` key.

    Returns
    -------
    ArrayLike
        The PSF kernel.
    """

    camera = jwst_data.camera.lower()
    filter = jwst_data.filter.lower()
    center_pixel = jwst_data.wcs.world_to_pixel(target_center)
    webbpsf_path = config_parameters.webbpsf_path
    psf_library_path = config_parameters.psf_library_path
    psf_arcsec = config_parameters.psf_arcsec
    normalize = config_parameters.normalize

    center_pixel_str = f"{int(10 * center_pixel[0])}p{int(10 * center_pixel[1])}"
    file_name = "_".join(
        (camera, filter, center_pixel_str, f"{psf_arcsec}arcsec", f"sub{subsample}")
    )
    path_to_file = join(psf_library_path, file_name)
    makedirs(psf_library_path, exist_ok=True)

    if isfile(path_to_file + ".npy"):
        logger.info(f"Loading PSF {file_name} from {psf_library_path}")
        return np.load(path_to_file + ".npy")

    psf = build_webb_psf(
        camera,
        filter,
        center_pixel,
        webbpsf_path,
        subsample,
        psf_arcsec,
        normalize,
    )

    logger.info(f"Saving PSF {file_name} in {psf_library_path}")

    np.save(path_to_file, psf)
    return psf
