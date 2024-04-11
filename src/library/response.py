from os.path import join

import numpy as np
import nifty8.re as jft
from .erosita_psf import eROSITA_PSF
from .data import Domain
from .utils import get_config

import jax.numpy as jnp
from jax import lax, vmap

from .erosita_observation import ErositaObservation


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
    if exposure_cut is not None:
        if exposure_cut < 0:
            raise ValueError("exposure_cut should be non-negative or None!")
        exposures[exposures < exposure_cut] = 0
    # FIXME short hack to remove additional axis. Also the Ifs should be restructed
    exposures = np.pad(exposures, ((0, 0), (43, 43), (43, 43)))
    return lambda x: exposures * x  # [np.newaxis, ...]


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


def _build_tm_erosita_psf(psf_filename, energy, pointing_center, domain, npatch,
                      margfrac, want_cut=False, convolution_method='LINJAX'):
    """
    Parameters:
    -----------
    psf_file: str
        filename tm_id_+ base / suffix, e.g. 2dpsf_190219v05.fits
    """
    psf = eROSITA_PSF(psf_filename)
    cdict = {"npatch": npatch,
             "margfrac": margfrac,
             "want_cut": want_cut}
    psf_func = psf.make_psf_op(energy, pointing_center, domain,
                               convolution_method, cdict)
    return psf_func


def build_erosita_psf(psf_filenames, energy, pointing_center, domain, npatch,
                      margfrac, want_cut=False, convolution_method='LINJAX'):

    functions = [_build_tm_erosita_psf(psf_file, energy, pcenter,
                                       domain, npatch, margfrac)
                 for psf_file, pcenter in zip(psf_filenames, pointing_center)]
    index = jnp.arange(len(functions))
    # FIXME make this more efficient
    vmap_functions = vmap(lambda i, x: lax.switch(i, functions, x), in_axes=(0, None))

    def vmap_psf_func(x):
        return vmap_functions(index, x)
    return vmap_psf_func

# FIXME only exposure 
def build_erosita_response(exposures, exposure_cut=0, tm_ids=None):
    # TODO: write docstring
    exposure = build_exposure_function(exposures, exposure_cut)
    mask = build_readout_function(exposures, exposure_cut, tm_ids)
    # psf = build_erosita_psf(-...)
    R = lambda x: mask(exposure(x))  # FIXME: should implement R = lambda x: mask(exposure(psf(x)))
    return R


def build_erosita_response_from_config(config_file_path):
    # TODO: write docstring
    # load config
    cfg = get_config(config_file_path)
    tel_info = cfg['telescope']
    file_info = cfg['files']
    psf_info = cfg['psf']

    # lists for exposure and psf files
    exposure_file_names = [join(file_info['obs_path'], f'{key}_'+file_info['exposure'])
                           for key in tel_info['tm_ids']]
    psf_file_names = [join(file_info['psf_path'], 'tm'+f'{key}_'+file_info['psf_base_filename'])
                      for key in tel_info['tm_ids']]

    # Get pointings for different telescope modules in RA/DEC
    obs_instance = ErositaObservation(file_info['input'],
                                      file_info['output'],
                                      file_info['obs_path'])
    center_stats = []
    for tm_id in tel_info['tm_ids']:
        tmp_center_stat = obs_instance.get_pointing_coordinates_stats(tm_id, file_info['input'])
        tmp_center_stat = [tmp_center_stat['RA'][0], tmp_center_stat['DEC'][0]]
        center_stats.append(tmp_center_stat)
    center_stats = np.array(center_stats)

    # center with respect to TM1
    ref_center = center_stats[0]
    d_centers = center_stats - ref_center
    # Set the Image pointing to the center and associate with TM1 pointing
    image_pointing_center = np.array(tuple([cfg['telescope']['fov']/2.]*2))
    pointing_center = d_centers + image_pointing_center
    domain = Domain(tuple([cfg['grid']['sdim']]*2), tuple([cfg['telescope']['fov']/cfg['grid']
    ['sdim']]*2))

    # get psf/exposure/mask function
    psf_func = build_erosita_psf(psf_file_names, psf_info['energy'], pointing_center, domain,
                                 psf_info['npatch'], psf_info['margfrac'], psf_info['want_cut'],
                                 psf_info['method'])

    exposure_func = build_callable_from_exposure_file(build_exposure_function,
                                                      exposure_file_names,
                                                      exposure_cut=tel_info['exp_cut'])

    mask_func = build_callable_from_exposure_file(build_readout_function,
                                                  exposure_file_names,
                                                  threshold=tel_info['exp_cut'],
                                                  keys=tel_info['tm_ids'])
    # plugin
    mask_func = lambda x: x
    exposure_func = lambda x: x
    psf_func = lambda x: x
    response_func = lambda x: mask_func(exposure_func(psf_func(x))[:,43:-43,43:-43])
    response_dict = {'mask': mask_func, 'exposure': exposure_func, 'psf': psf_func,
                     'R': response_func}
    return response_dict


def load_erosita_response():
    pass  # FIXME: implement
