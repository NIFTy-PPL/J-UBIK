import numpy as np
import nifty8.re as jft

from .utils import chain_callables


def build_exposure_function(exposures, exposure_cut=None):
    """
    Returns a function that applies instrument exposures to an input array.

    Parameters
    ----------
    exposures : ndarray
        Array with instrument exposure maps. The 0-th axis indexes the telescope module (for
        multi-module instruments).
    exposure_cut : float or None, optional
        A threshold exposure value below which exposures are set to zero.
        If None (default), no threshold is applied.

    Returns
    -------
    callable
        A function that takes an input array `x` and returns the element-wise
        product of `exposures` and `x`, with the first dimension of `exposures`
        broadcasted to match the shape of `x`.

    Raises
    ------
    ValueError:
        If `exposures` is not a 2D array or `exposure_cut` is negative.
    """
    if exposure_cut is not None and exposure_cut < 0:
        raise ValueError("exposure_cut should be positive or None!")
    if exposure_cut is not None:
        exposures[exposures < exposure_cut] = 0
    return lambda x: exposures * x[np.newaxis, ...]


def build_exposure_readout_function(exposures, exposure_cut=None, keys=None):
    """
    Applies a readout corresponding to the exposure masks.

    Parameters
    ----------
        exposures : ndarray
        Array with instrument exposure maps. The 0-th axis indexes the telescope module (for
        multi-module instruments).
        exposure_cut: float or None, optional
            A threshold exposure value below which exposures are set to zero.
            If None (default), no threshold is applied.
        keys : tuple or list or None
            A tuple containing the ids of the telescope modules to be used as keys for the
            response output dictionary. Optional for a single module observation.
    Returns
    -------
        function: A callable that applies an exposure mask to an input sky.
    Raises:
    -------
        ValueError:
        If exposure_cut is negative.
        If keys does not have the right shape.
        If the exposures do not have the right shape.
    """
    if exposure_cut < 0:
        raise ValueError("exposure_cut should be positive!")
    if exposure_cut is not None:
        exposures[exposures < exposure_cut] = 0
    mask = exposures == 0
    if keys is None:
        keys = ['masked input']
    elif len(keys) != exposures.shape[0]:
        raise ValueError("length of keys should match the number of exposure maps.")

    def _apply_readout(exposured_sky: np.array):
        if len(mask.shape) != 3:
            raise ValueError("exposures should have shape (n, m, q)!")
        return jft.Vector({key: exposured_sky[i][~mask[i]] for i, key in enumerate(keys)})

    return _apply_readout


def build_callable_from_exposure_file(callable, exposure_filenames, **kwargs):
    """
    Returns a callable function which is built from a NumPy array of exposures loaded from file.

    Parameters
    ----------
    callable : function
        A callable function that takes a NumPy array of exposures as input.
    exposure_filenames : list[str]
        A list of filenames of exposure files to load.
        Files should be in a .npy or .fits format.
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the callable function.

    Returns
    -------
    result : object
        The result of applying the callable function to the loaded exposures.

    Raises
    ------
    ValueError:
        If any of the exposure files are not in a .npy or .fits format.

    Notes
    -----
    This function loads exposure files from disk and applies a callable function to the loaded
    exposures. The exposure files should be in a .npy or .fits format. The loaded exposures are
    stored in a NumPy array, which is passed as input to the callable function. Additional
    keyword arguments can be passed to the callable function using **kwargs. The result of
    applying the callable function to the loaded exposures is returned as output.
    """
    if not isinstance(exposure_filenames, list):
        raise ValueError('`exposure_filenames` should be a `list`.')
    exposures = []
    for file in exposure_filenames:
        if file.endswith('.npy'):
            exposures.append(np.load(file))
        elif file.endswith('.fits'):
            from astropy.io import fits
            exposures.append(fits.open(file)[0].data)
        elif not (file.endswith('.npy') or file.endswith('.fits')):
            raise ValueError('exposure files should be in a .npy or .fits format!')
        else:
            raise FileNotFoundError(f'cannot find {file}!')
    exposures = np.array(exposures)
    return callable(exposures, **kwargs)


def build_erosita_psf(psf_shape, tm_ids, energy, center, convolution_method):
    pass  # FIXME: implement


def build_erosita_psf_from_file(exposure_filenames, exposure_cut, tm_ids):
    pass  # FIXME: implement


def build_erosita_response(exposures, exposure_cut, tm_ids):
    # TODO: write docstring
    exposure = build_exposure_function(exposures, exposure_cut)
    mask = build_exposure_readout_function(exposures, exposure_cut, tm_ids)
    R = chain_callables(mask, exposure)  # FIXME: should implement R = mask @ exposure @ conv_op
    return R



def build_erosita_response_from_config(config_file):
    pass  # FIXME: implement


def load_erosita_response():
    pass  # FIXME: implement
