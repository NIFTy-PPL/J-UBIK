import numpy as np
import nifty8.re as jft


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


def build_readout_function(flasgs, threshold=None, keys=None):
    """
    Applies a readout corresponding to input flags.

    Parameters
    ----------
        flags : ndarray
        Array with flags. Where flags are equal to zero the input will not be read out.
        The 0-th axis indexes the number of 2D flag maps, e.g. it could index the telescope module
        (for multi-module instruments exposure maps).
        threshold: float or None, optional
            A threshold value below which flags are set to zero (e.g., an exposure cut).
            If None (default), no threshold is applied.
        keys : list or tuple or None
            A list or tuple containing the keys for the response output dictionary.
            For example, a list of the telescope modules ids for a multi-module instrument.
            Optional for a single-module observation.
    Returns
    -------
        function: A callable that applies a mask to an input array (e.g. an input sky) and returns
        a `nifty8.re.Vector` containing a dictionary of read-out inputs.
    Raises:
    -------
        ValueError:
        If threshold is negative.
        If keys does not have the right shape.
        If the flags do not have the right shape.
    """
    if threshold < 0:
        raise ValueError("threshold should be positive!")
    if threshold is not None:
        flasgs[flasgs < threshold] = 0
    mask = flasgs == 0
    if keys is None:
        keys = ['masked input']
    elif len(keys) != flasgs.shape[0]:
        raise ValueError("length of keys should match the number of flag maps.")

    def _apply_readout(x: np.array):
        """
        Reads out input array (e.g, sky signals to which an exposure is applied)
        at locations specified by a mask.
        Args:
            x: ndarray

        Returns:
            readout = `nifty8.re.Vector` containing a dictionary of read-out inputs.
        """
        if len(mask.shape) != 3:
            raise ValueError("flags should have shape (n, m, q)!")
        return jft.Vector({key: x[i][~mask[i]] for i, key in enumerate(keys)})

    return _apply_readout


def build_callable_from_exposure_file(builder, exposure_filenames, **kwargs):
    """
    Returns a callable function which is built from a NumPy array of exposures loaded from file.

    Parameters
    ----------
    builder : function
        A builder function that takes a NumPy array of exposures as input and outputs a callable.
        The callable should perform an operation on a different object using the exposure.
    exposure_filenames : list[str]
        A list of filenames of exposure files to load.
        Files should be in a .npy or .fits format.
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the callable function.

    Returns
    -------
    result : callable
        The callable function built with the exposures loaded from file.

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
    exposures = np.array(exposures, dtype="float64") # in fits there is only >f4 meaning float32
    return builder(exposures, **kwargs)


def build_erosita_psf(psf_shape, tm_ids, energy, center, convolution_method):
    pass  # FIXME: implement


def build_erosita_response(exposures, exposure_cut=0, tm_ids=None):
    # TODO: write docstring
    exposure = build_exposure_function(exposures, exposure_cut)
    mask = build_readout_function(exposures, exposure_cut, tm_ids)
    # psf = build_erosita_psf(-...)
    R = lambda x: mask(exposure(x))  # FIXME: should implement R = lambda x: mask(exposure(psf(x)))
    return R


def build_erosita_response_from_config(config_file):
    #op = build_erosita_response(*config_file)
    pass  # FIXME: implement


def load_erosita_response():
    pass  # FIXME: implement
