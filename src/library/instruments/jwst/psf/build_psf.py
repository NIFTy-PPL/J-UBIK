import numpy as np
from jax.scipy.signal import fftconvolve
from functools import partial
from os.path import join, isfile

from typing import Tuple, Callable
from numpy.typing import ArrayLike


def load_psf_kernel(
    camera: str,
    filter: str,
    center_pixel: Tuple[float],
    webbpsf_path: str,
    psf_library_path: str,
    fov_pixels: int,
    subsample: int,
    normalize: str = 'last'
) -> ArrayLike:

    camera = camera.lower()
    filter = filter.lower()
    center_pixel_str = f'{int(10*center_pixel[0])}p{int(10*center_pixel[1])}'
    fov_pixel_str = f'{fov_pixels}'
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
        fov_pixels,
        subsample,
        normalize)

    print('*'*80)
    print(f'Saving {file_name} in {psf_library_path}')
    print('*'*80)

    np.save(path_to_file, psf)
    return psf


def build_webb_psf(
    camera: str,
    filter: str,
    center_pixel: Tuple[float],
    webbpsf_path: str,
    fov_pixels: int,
    subsample: int,
    normalize: str = 'last'
):
    import webbpsf
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
        # fov_arcsec=fov_arcsec,
        oversample=subsample,
        normalize='last')

    return psf[2].data


def PsfOperator_fft(field, kernel):
    '''Creates a Psf-operator: convolution of field by kernel'''
    return fftconvolve(field, kernel, mode='same')


def instantiate_psf(psf: ArrayLike | None) -> Callable[ArrayLike, ArrayLike]:
    '''Build psf convolution operator from psf array. If None is provided the
    psf operator returns the field.

    Parameters
    ----------
    psf : ArrayLike or None

    Return
    ------
    Callable which convolves its input by the psf kernel.
    If psf kernel is None, the Callable is lambda x: x
    '''
    if psf is None:
        return lambda x: x

    return partial(PsfOperator_fft, kernel=psf)
