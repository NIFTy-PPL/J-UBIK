import jax.numpy as jnp
import numpy as np


def apply_erosita_exposure(exposures, exposure_cut=None):
    """
    Returns a function that applies eROSITA exposures to an input array.

    Parameters
    ----------
    exposures: ndarray
    Array with eROSITA exposure maps. The 0-th axis indexes the telescope module.
    exposure_cut: float or None, optional
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
    ValueError
        If `exposures` is not a 2D array or `exposure_cut` is negative.
    """
    if exposure_cut < 0:
        raise ValueError("exposure_cut should be positive!")
    if exposure_cut is not None:
        exposures[exposures < exposure_cut] = 0
    return lambda x: exposures * x[jnp.newaxis, ...]


def apply_erosita_exposure_from_file(exposure_filenames, exposure_cut=None):
    """
    Returns a function that applies eROSITA exposures loaded from files to an input array.

    Parameters
    ----------
    exposure_filenames : list[str]
        A list of file names containing eROSITA exposure maps in numpy format.
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
    ValueError
        If any of the exposure files cannot be loaded or `exposure_cut` is negative.
    """
    exposures = []
    for file in exposure_filenames:
        exposures.append(np.load(file))
    exposures = np.array(exposures)
    return apply_erosita_exposure(exposures, exposure_cut)


def apply_erosita_psf(shape, tm_ids, energy, center, convolution_method):
    pass  # FIXME: implement


def apply_erosita_psf_from_file():
    pass  # FIXME: implement


# def apply_erosita_response(x, fov, emin, emax, npix, tm_ids):
#
#
#     R = mask @ sky_model.pad.adjoint @ exposure_op @ conv_op
#     return R

def apply_erosita_readout(exposures, exposure_cut, tm_ids):
    """
    Applies a readout corresponding to the exposure masks.

    Parameters
    ----------
        exposures : ndarray
        Array with eROSITA exposure maps. The 0-th axis indexes the telescope module.
        exposure_cut: float or None, optional
            A threshold exposure value below which exposures are set to zero.
            If None (default), no threshold is applied.
        tm_ids : tuple or list
            A tuple containing the ids of the telescope modules to be used as keys for the
            response output dictionary
    Returns
    -------
        function: A lambda function that extracts data from the exposures array
            based on the tm_ids values.
    Raises:
    -------
        ValueError: If exposure_cut is negative.
    """
    if exposure_cut < 0:
        raise ValueError("exposure_cut should be positive!")
    if exposure_cut is not None:
        exposures[exposures < exposure_cut] = 0
    mask = exposures == 0
    return lambda x: {key: x[i][~mask[i]] for i, key in enumerate(tm_ids)}


def apply_erosita_readout_from_file(exposure_filenames, exposure_cut, tm_ids):
    """
    Applies a readout corresponding to the exposure masks from file.

    Parameters
    ----------
        exposure_filenames : ndarray
        A list of file names containing eROSITA exposure maps in numpy format.
        exposure_cut: float or None, optional
            A threshold exposure value below which exposures are set to zero.
            If None (default), no threshold is applied.
        tm_ids : tuple or list
            A tuple containing the ids of the telescope modules to be used as keys for the
            response output dictionary
    Returns
    -------
        function: A lambda function that extracts data from the exposures array
            based on the tm_ids values.
    Raises:
    -------
        ValueError: If exposure_cut is negative.
    """
    exposures = []
    for file in exposure_filenames:
        exposures.append(np.load(file))
    exposures = np.array(exposures)
    return apply_erosita_readout(exposures, exposure_cut, tm_ids)


def apply_erosita_response_from_config():
    pass


def load_erosita_response():
    pass
