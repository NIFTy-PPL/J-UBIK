# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %
from .parse.jwst_psf_parse import PsfKernelModel

from functools import partial
from os.path import join, isfile
from typing import Tuple, Callable, Optional

import numpy as np
from jax.scipy.signal import fftconvolve
from numpy.typing import ArrayLike


def build_webb_psf(
    camera: str,
    filter: str,
    center_pixel: Tuple[float],
    webbpsf_path: str,
    subsample: int,
    fov_pixels: Optional[int] = None,
    fov_arcsec: Optional[float] = None,
    normalize: str = 'last'
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
    fov_pixels : int, optional
        The field of view (FOV) in pixels.
        If provided, it specifies the size of the PSF image in
        pixels. If not provided, `fov_arcsec` must be specified.
    fov_arcsec : float, optional
        The field of view (FOV) in arcseconds.
        If provided, it specifies the size of the PSF image
        in arcseconds. If not provided, `fov_pixels` must be specified.
    normalize : str, optional
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
    ValueError
        If neither `fov_pixels` nor `fov_arcsec` is provided.
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
    if fov_pixels is None and fov_arcsec is None:
        raise ValueError('You need to provide either fov_pixels or fov_arcsec')

    try:
        import webbpsf
    except ImportError:
        raise ImportError("webbpsf is not installed. Please install it first")
    from os import environ
    environ["WEBBPSF_PATH"] = webbpsf_path

    psf_model_supported = {
        'nircam': webbpsf.NIRCam(),
        'miri': webbpsf.MIRI(),
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
        fov_pixels=fov_pixels,
        fov_arcsec=fov_arcsec,
        oversample=subsample,
        normalize=normalize)

    return psf[2].data


def load_psf_kernel(
    camera: str,
    filter: str,
    center_pixel: Tuple[float],
    webbpsf_path: str,
    psf_library_path: str,
    subsample: int,
    psf_pixels: Optional[int] = None,
    psf_arcsec: Optional[float] = None,
    normalize: str = 'last'
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
    camera : str
        The camera model for which to compute the PSF.
        Options are 'nircam' and 'miri'.
        This value is converted to lowercase before processing.
    filter : str
        The filter for which to compute the PSF.
    center_pixel : tuple of float
        The (x, y) coordinates of the center pixel for the PSF calculation.
    webbpsf_path : str
        The path to the directory containing the `webbpsf` data files.
    psf_library_path : str
        The directory where the computed PSF files are stored.
        The PSF kernel will be saved here if it is not already present.
    subsample : int
        The oversampling factor for the PSF computation.
    psf_pixels : int, optional
        The size of the PSF evaluation in pixels. If not provided, `psf_arcsec`
        must be specified.
    psf_arcsec : float, optional
        The size of the PSF evaluation in arcsec. If not provided, `psf_pixels`
        must be specified.
    normalize : str, optional
        The normalization method for the PSF. Default is 'last',
        but other methods may be supported by `webbpsf`.

    Returns
    -------
    numpy.ndarray
        The PSF data as a 2D array.

    Raises
    ------
    ValueError
        If neither `psf_pixels` nor `psf_arcsec` is provided.
    """
    if psf_pixels is None and psf_arcsec is None:
        raise ValueError('You need to provide either psf_pixels or psf_arcsec')
    elif psf_pixels is not None and psf_arcsec is not None:
        print('Both psf_pixels and psf_arcsec are provided.'
              f'Using psf_arcsec: {psf_arcsec}.')
        psf_pixels = None

    camera = camera.lower()
    filter = filter.lower()
    center_pixel_str = f'{int(10*center_pixel[0])}p{int(10*center_pixel[1])}'
    fov_pixel_str = f'{psf_pixels}' if psf_pixels is not None else f'{psf_arcsec}arcsec'
    file_name = '_'.join(
        (camera, filter, center_pixel_str, fov_pixel_str))
    path_to_file = join(psf_library_path, file_name)

    if isfile(path_to_file + '.npy'):
        print('*'*80)
        print(f'Loading {file_name} from {psf_library_path}')
        print('*'*80)
        return np.load(path_to_file + '.npy')

    psf = build_webb_psf(
        camera,
        filter,
        center_pixel,
        webbpsf_path,
        subsample,
        psf_pixels,
        psf_arcsec,
        normalize)

    print('*'*80)
    print(f'Saving {file_name} in {psf_library_path}')
    print('*'*80)

    np.save(path_to_file, psf)
    return psf


def load_psf_kernel_from_psf_kernel_model(
    psf_kernel_model: Optional[PsfKernelModel]
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
    psf_kernel_model: PsfKernelModel
        - camera : str
        - filter : str
        - center_pixel : tuple of float
        - webbpsf_path : str
        - psf_library_path : str
        - subsample : int
        - psf_arcsec : float, optional
        - normalize : str, optional

    Returns
    -------
    numpy.ndarray
        The PSF data as a 2D array.
    """
    if psf_kernel_model is None:
        return None

    return load_psf_kernel(
        camera=psf_kernel_model.camera,
        filter=psf_kernel_model.filter,
        center_pixel=psf_kernel_model.pointing_center,
        subsample=psf_kernel_model.subsample,
        webbpsf_path=psf_kernel_model.config_parameters.webbpsf_path,
        psf_library_path=psf_kernel_model.config_parameters.psf_library_path,
        psf_arcsec=psf_kernel_model.config_parameters.psf_arcsec,
    )


def psf_operator_fft(field, kernel):
    """
    Apply a Point Spread Function (PSF) operator using FFT-based convolution.

    This function performs the convolution of an input field (e.g., an image)
    with a PSF kernel using FFT (Fast Fourier Transform) for efficient
    computation.
    The convolution is done in 'same' mode, meaning the output array has the
    same shape as the input field.

    Parameters
    ----------
    field : np.ndarray
        The input 2D array (e.g., an image or field) to be convolved with
        the kernel.
    kernel : np.ndarray
        The PSF kernel to apply to the field. It should have the same or
        compatible dimensions as the field.

    Returns
    -------
    np.ndarray
        The resulting field after convolving with the PSF kernel.
        This will have the same shape as the input field.
    """
    return fftconvolve(field, kernel, mode='same')


def instantiate_psf(psf: ArrayLike | None) -> Callable[[ArrayLike], ArrayLike]:
    """Build psf convolution operator from psf array. If None is provided the
    psf operator returns the field.

    Parameters
    ----------
    psf : ArrayLike or None

    Return
    ------
    Callable which convolves its input by the psf kernel.
    If psf kernel is None, the Callable is lambda x: x
    """
    if psf is None:
        return lambda x: x

    return partial(psf_operator_fft, kernel=psf)
