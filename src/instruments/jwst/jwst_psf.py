# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %
from .jwst_data import JwstData
from .parse.jwst_psf import PsfKernelConfig

from functools import partial
from os.path import join, isfile
from typing import Tuple, Callable


import numpy as np
from jax.scipy.signal import fftconvolve
from numpy.typing import ArrayLike
from astropy.coordinates import SkyCoord


def build_webb_psf(
    camera: str,
    filter: str,
    center_pixel: Tuple[float],
    webbpsf_path: str,
    subsample: int,
    fov_arcsec: float | None = None,
    normalize: str = "last",
):
    """
    Builds a Point Spread Function (PSF) model for the JWST using the specified
    camera and filter.

    This function computes the PSF using the `webbpsf` library.
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
        The file path to the directory where the `webbpsf` data is stored.
        This directory is
        used to load the PSF models.
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
        last computed value. Other options may be supported by `webbpsf`.

    Returns
    -------
    numpy.ndarray
        The computed PSF data as a 2D array.

    Raises
    ------
    ImportError
        If `webbpsf` is not installed.
    KeyError
        If the specified `camera` is not supported by the `webbpsf` library.

    Notes
    -----
    - Ensure that the `webbpsf` library is properly installed and that `
      webbpsf_path` is correctly set to point to the location of the JWST
      PSF data files.
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
        import webbpsf
    except ImportError:
        raise ImportError("webbpsf is not installed. Please install it first")
    from os import environ

    environ["WEBBPSF_PATH"] = webbpsf_path

    psf_model_supported = {
        "nircam": webbpsf.NIRCam(),
        "miri": webbpsf.MIRI(),
    }

    try:
        psf_model = psf_model_supported[camera.lower()]
    except KeyError as ke:
        raise KeyError(
            f"You requested {camera.lower()} psf model."
            f"Supported psf models: {psf_model_supported.keys()}"
        ) from ke

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
    target_center: SkyCoord,
    config_parameters: PsfKernelConfig,
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
    subsample = jwst_data.meta.subsample
    webbpsf_path = config_parameters.webbpsf_path
    psf_library_path = config_parameters.psf_library_path
    psf_arcsec = config_parameters.psf_arcsec
    normalize = config_parameters.normalize

    center_pixel_str = f"{int(10 * center_pixel[0])}p{int(10 * center_pixel[1])}"
    file_name = "_".join((camera, filter, center_pixel_str, f"{psf_arcsec}arcsec"))
    path_to_file = join(psf_library_path, file_name)

    if isfile(path_to_file + ".npy"):
        print("*" * 80)
        print(f"Loading {file_name} from {psf_library_path}")
        print("*" * 80)
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

    print("*" * 80)
    print(f"Saving {file_name} in {psf_library_path}")
    print("*" * 80)

    np.save(path_to_file, psf)
    return psf


def build_psf_operator(
    psf_kernel: ArrayLike | None,
) -> Callable[[ArrayLike], ArrayLike] | None:
    """Build psf convolution operator from psf kernel array.
    If None is provided the psf operator returns the field.

    Parameters
    ----------
    psf_kernel : ArrayLike or None
        The psf kernel.

    Returns
    -------
    Callable which convolves its input by the psf kernel.
    If psf kernel is None, the Callable is lambda x: x
    """
    if psf_kernel is None:
        return lambda x: x

    return partial(fftconvolve, in2=psf_kernel, mode="same")
