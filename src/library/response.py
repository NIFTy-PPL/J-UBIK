from os.path import join, basename
from pathlib import Path

import numpy as np
import nifty8.re as jft
from astropy.io import fits

from .erosita_psf import eROSITA_PSF
from .data import Domain
from .utils import get_config

import jax.numpy as jnp
from jax import lax, vmap

from .erosita_observation import ErositaObservation
from ..operators.convolve_utils import linpatch_convolve


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
    return lambda x: exposures * x  # [np.newaxis, ...]


def build_readout_function(flasgs, threshold=None, keys=None):
    """
    Applies a readout corresponding to input flags.

    Parameters
    ----------
        flags : ndarray
        Array with flags. Where flags are equal to zero the input will not be read out.
        The 0-th axis indexes the number of 3D flag maps, e.g. it could index the telescope module
        (for multi-module instruments exposure maps).
        The 1st axis indexes the energy direction.
        The 2nd and 3rd axis refer to the spatial direction.
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
        raise ValueError(
            "length of keys should match the number of flag maps.")

    def _apply_readout(x: np.array):
        """
        Reads out input array (e.g, sky signals to which an exposure is applied)
        at locations specified by a mask.
        Args:
            x: ndarray

        Returns:
            readout = `nifty8.re.Vector` containing a dictionary of read-out inputs.
        """
        if len(mask.shape) != 4:
            raise ValueError("flags should have shape (n, m, q, l)!")

        if len(x.shape) != 4:
            raise ValueError("input should have shape (n, m, q, l)!")
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
        The filenames should begin with "tm" followed by the index of the telescope module,
        e.g. "tm1".
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
    tm_id = basename(exposure_filenames[0])[2]
    tm_exposures = []
    for file in exposure_filenames:
        if basename(file)[2] != tm_id:
            exposures.append(tm_exposures)
            tm_exposures = []
            tm_id = basename(file)[2]
        if basename(file)[2] == tm_id:
            if file.endswith('.npy'):
                tm_exposures.append(np.load(file))
            elif file.endswith('.fits'):
                tm_exposures.append(fits.open(file)[0].data)
            elif not (file.endswith('.npy') or file.endswith('.fits')):
                raise ValueError(
                    'exposure files should be in a .npy or .fits format!')
            else:
                raise FileNotFoundError(f'cannot find {file}!')
    exposures.append(tm_exposures)
    # in fits there is only >f4 meaning float32
    exposures = np.array(exposures, dtype="float64")
    return builder(exposures, **kwargs)


def calculate_erosita_effective_area(path_to_caldb, tm_ids, e_min, e_max,
                                     caldb_folder_name='caldb',
                                     arf_filename_suffix='_arf_filter_000101v02.fits',
                                     n_points=500):
    """
    Returns the effective area for the given energy range (in keV) and
    telescope module list (in cm^2).
    The effective area is computed by linearly interpolating the effective area contained in the ARF
     file on the desired energy ranges and taking the average within each energy range.

    Parameters
    ----------
    path_to_caldb : str
        Path to the eROSITA calibration database.
    tm_ids : list
        List of telescope module IDs.
    e_min : np.ndarray
        Minimum energies (in keV) at which to calculate the effective area.
    e_max : np.ndarray
        Maximum energies (in keV) at which to calculate the effective area.
    caldb_folder_name : str, optional
        Name of the calibration database folder.
    arf_filename_suffix : str, optional
        Suffix of the ARF file name.
    n_points : int, optional
        Number of points to use for the interpolation.

    Returns
    -------
    effective_areas : np.ndarray
        Effective area in cm^2

    Raises
    ------
    ValueError
        If the energy is out of range.
    """
    path = join(path_to_caldb, caldb_folder_name, 'data', 'erosita')
    effective_areas = []
    e_min = np.array(e_min)
    e_max = np.array(e_max)
    for tm_id in tm_ids:
        arf_filename = join(path, f'tm{tm_id}', 'bcf', f'tm{tm_id}{arf_filename_suffix}')
        with fits.open(arf_filename) as f:
            energy_low = f['SPECRESP'].data["ENERG_LO"]
            energy_hi = f['SPECRESP'].data["ENERG_HI"]
            effective_area = f['SPECRESP'].data["SPECRESP"]
        if np.any(e_min < energy_low[0]):
            loc = np.where(e_min < energy_low[0])
            raise ValueError(f'Energy {e_min[loc]} keV is out of range!')
        if np.any(e_max > energy_hi[-1]):
            loc = np.where(e_max > energy_hi[-1])
            raise ValueError(f'Energy {e_max[loc]} keV is out of range!')
        coords = (energy_hi + energy_low) / 2
        effective_areas_in_bin = []
        for i in range(e_min.shape[0]):
            energies = np.linspace(e_min[i], e_max[i], n_points)
            effective_areas_in_bin.append(np.mean(np.interp(energies, coords, effective_area)))
        effective_areas.append(effective_areas_in_bin)
    return np.array(effective_areas)


def _build_tm_erosita_psf(psf_filename, energies, pointing_center, domain,
                          npatch, margfrac, want_cut=False,
                          convolution_method='LINJAX'):
    """
    #TODO only brief docstring, not public.

    Parameters:
    -----------
    psf_file: str
        filename tm_id_+ base / suffix, e.g. 2dpsf_190219v05.fits
    energies: list
    """
    psf = eROSITA_PSF(psf_filename)
    cdict = {"npatch": npatch,
             "margfrac": margfrac,
             "want_cut": want_cut}
    psf_func = psf.make_psf_op(energies,
                               pointing_center,
                               domain,
                               convolution_method,
                               cdict)
    return psf_func


def _build_tm_erosita_psf_array(psf_filename, energies, pointing_center,
                                domain, npatch):
    """
    #TODO only brief docstring, not public.

    Parameters:
    -----------
    psf_file: str
        filename tm_id_+ base / suffix, e.g. 2dpsf_190219v05.fits
    energies: list
    """
    psf = eROSITA_PSF(psf_filename)
    psf_array = psf.make_interpolated_psf_array(energies,
                                                pointing_center,
                                                domain,
                                                npatch)
    return psf_array


def build_erosita_psf(psf_filenames, energies, pointing_center,
                      domain, npatch, margfrac):
    """
    Parameters:
    ----------
    psf_filenames: list(str), path to psf files from calibration
    energies: list
    pointing_center: list(float) #TODO Check types
    domain: ju.Domain
    npatch: int, number of patches
    margfrac: margin fraction, fractional size of the margin. margin/inputsize.
            Needed to break the periodic boundary conditions (PBC) in the patch
            convolution.
    """
    psfs = [_build_tm_erosita_psf_array(psf_file, energies, pcenter,
                                        domain, npatch)
            for psf_file, pcenter in zip(psf_filenames, pointing_center)]
    psfs = np.array(psfs)

    # FIXME Check sqrt npatches etc

    shp = (domain.shape[-2], domain.shape[-1])
    margin = max((int(np.ceil(margfrac*ss)) for ss in shp))

    def psf_op(x):
        return vmap(linpatch_convolve, in_axes=(None, None, 0, None, None))(x, domain, psfs, npatch, margin)

    return psf_op


# func = lambda psf_file,x,y,z: build_psf(psf_file,x, y, z)
#     vmap_func = jax.vmap(func)(psf_file, x, y, z)
#     vmap_func(x)


def build_erosita_response(exposures, exposure_cut=0, tm_ids=None):
    # TODO: write docstring
    exposure = build_exposure_function(exposures, exposure_cut)
    mask = build_readout_function(exposures, exposure_cut, tm_ids)
    # psf = build_erosita_psf(-...)
    # FIXME: should implement R = lambda x: mask(exposure(psf(x)))
    def R(x): return mask(exposure(x))
    return R


def build_erosita_response_from_config(config_file_path):
    """ Builds the eROSITA response from a yaml config file.
    #TODO Example / Needed Entries in yaml
    """
    # load config
    cfg = get_config(config_file_path)
    tel_info = cfg['telescope']
    file_info = cfg['files']
    psf_info = cfg['psf']
    grid_info = cfg['grid']

    # load energies
    e_min = grid_info['energy_bin']['e_min']
    e_max = grid_info['energy_bin']['e_max']

    if not isinstance(e_min, list):
        raise TypeError("e_min must be a list!")

    if not isinstance(e_max, list):
        raise TypeError("e_max must be a list!")

    if len(e_max) != len(e_max):
        raise ValueError("e_min and e_max must have the same length!")

    # lists for exposure and psf files
    exposure_filenames = []
    for tm_id in tel_info['tm_ids']:
        exposure_filename = f'tm{tm_id}_' + file_info['exposure']
        [exposure_filenames.append(join(file_info['obs_path'],
                                        "processed",
                                        f"{Path(exposure_filename).stem}_emin{e}_emax{E}.fits"))
         for e, E in zip(e_min, e_max)]

    psf_file_names = [join(file_info['psf_path'], 'tm'+f'{key}_'+file_info['psf_base_filename'])
                      for key in tel_info['tm_ids']]

    # Get pointings for different telescope modules in RA/DEC
    obs_instance = ErositaObservation(file_info['input'],
                                      file_info['output'],
                                      file_info['obs_path'])
    center_stats = []
    for tm_id in tel_info['tm_ids']:
        tmp_center_stat = obs_instance.get_pointing_coordinates_stats(
            tm_id, file_info['input'])
        tmp_center_stat = [tmp_center_stat['RA'][0], tmp_center_stat['DEC'][0]]
        center_stats.append(tmp_center_stat)
    center_stats = np.array(center_stats)

    # center with respect to TM1
    ref_center = center_stats[0]
    d_centers = center_stats - ref_center
    # Set the Image pointing to the center and associate with TM1 pointing
    image_pointing_center = np.array(tuple([cfg['telescope']['fov']/2.]*2))
    pointing_center = d_centers + image_pointing_center

    # FIXME distances for domain energy shouldn't be hardcoded 1
    domain = Domain(tuple([cfg['grid']['edim']] + [cfg['grid']['sdim']]*2),
                    tuple([1]+[cfg['telescope']['fov']/cfg['grid']['sdim']]*2))

    # get psf/exposure/mask function
    psf_func = build_erosita_psf(psf_file_names, psf_info['energy'], pointing_center,
                                 domain, psf_info['npatch'], psf_info['margfrac'])

    tmp = build_callable_from_exposure_file(build_exposure_function,
                                            exposure_filenames,
                                            exposure_cut=tel_info['exp_cut'])

    if tel_info['effective_area_correction']:
        caldb_folder_name = 'caldb'
        arf_filename_suffix = '_arf_filter_000101v02.fits'
        if 'caldb_folder_name' in file_info.keys():
            caldb_folder_name = file_info['caldb_folder_name']
        if 'arf_filename_suffix' in file_info.keys():
            arf_filename_suffix = file_info['arf_filename_suffix']
        effective_area = calculate_erosita_effective_area(file_info['calibration_path'],
                                                          tel_info['tm_ids'],
                                                          np.array(e_min),
                                                          np.array(e_max),
                                                          caldb_folder_name=caldb_folder_name,
                                                          arf_filename_suffix=arf_filename_suffix)
        exposure_func = lambda x: tmp(x) * effective_area[:, :, np.newaxis, np.newaxis]
    else:
        exposure_func = tmp

    mask_func = build_callable_from_exposure_file(build_readout_function,
                                                  exposure_filenames,
                                                  threshold=tel_info['exp_cut'],
                                                  keys=tel_info['tm_ids'])

    # plugin
    def response_func(x): return mask_func(exposure_func(psf_func(x)))
    response_dict = {'mask': mask_func, 'exposure': exposure_func, 'psf': psf_func,
                     'R': response_func}
    return response_dict


def load_erosita_response():
    pass  # FIXME: implement
