from os.path import basename, join
from pathlib import Path

import numpy as np
from astropy.io import fits
from jax import vmap

from .erosita_psf import eROSITA_PSF
from ...data import Domain
from ...erosita_observation import ErositaObservation
from ...response import build_exposure_function, build_readout_function
from ...utils import get_config
from ....operators.jifty_convolution_operators import linpatch_convolve


def build_callable_from_exposure_file(builder, exposure_filenames, **kwargs):
    """
    Returns a callable function which is built from a NumPy array of
    exposures loaded from file.

    Parameters
    ----------
    builder : function
        A builder function that takes a NumPy array of exposures as input and
        outputs a callable.
        The callable should perform an operation on a different object using
        the exposure.
    exposure_filenames : list[str]
        A list of filenames of exposure files to load.
        Files should be in a .npy or .fits format.
        The filenames should begin with "tm" followed by the index of the
        telescope module,
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
    This function loads exposure files from disk and applies a callable
    function to the loaded
    exposures. The exposure files should be in a .npy or .fits format. The
    loaded exposures are
    stored in a NumPy array, which is passed as input to the callable
    function. Additional
    keyword arguments can be passed to the callable function using **kwargs.
    The result of
    applying the callable function to the loaded exposures is returned as
    output.
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
    exposures = np.array(exposures,
                         dtype="float64")  # in fits there is only >f4
    # meaning float32
    return builder(exposures, **kwargs)


def calculate_erosita_effective_area(path_to_caldb, tm_ids, e_min, e_max,
                                     caldb_folder_name='caldb',
                                     arf_filename_suffix='_arf_filter_000101v02.fits',
                                     n_points=500):
    """
    Returns the effective area for the given energy range (in keV) and
    telescope module list (in cm^2).
    The effective area is computed by linearly interpolating the effective
    area contained in the ARF
     file on the desired energy ranges and taking the average within each
     energy range.

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
        arf_filename = join(path, f'tm{tm_id}', 'bcf',
                            f'tm{tm_id}{arf_filename_suffix}')
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
            effective_areas_in_bin.append(
                np.mean(np.interp(energies, coords, effective_area)))
        effective_areas.append(effective_areas_in_bin)
    return np.array(effective_areas)


def _build_tm_erosita_psf(psf_filename, energies, pointing_center, domain,
                          npatch, margfrac, want_cut=False,
                          convolution_method='LINJAX'):
    """
    Creates a point spread function (PSF) operator for eROSITA using the
    provided
    PSF file and parameters.

    Parameters:
    -----------
    psf_filename : str
        Filename of the PSF file, e.g., '2dpsf_190219v05.fits'.
    energies : list of float
        List of energies in keV for which the PSF will be computed.
    pointing_center : array-like
        The pointing center coordinates for the PSF.
    domain : object
        The domain over which the PSF will be defined.
    npatch : int
        Number of patches in the PSF.
    margfrac : float
        Fraction of margin to be considered for the linpatch convolution.
    want_cut : bool, optional
        If True, apply a cut to the PSF. Default is False.
    convolution_method : str, optional
        Method for convolution, default is 'LINJAX', which corresponds
        to the linpatch convolution method.

    Returns:
    --------
    psf_func : callable
        A function representing the PSF operator.
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
    Builds the PSF array for eROSITA using the provided PSF file and parameters.
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
    Constructs a PSF (Point Spread Function) operator for the eROSITA telescope
    using a set of PSF files and associated parameters.

    Parameters:
    ----------
    psf_filenames : list of str
        Paths to PSF files obtained from calibration, e.g., ['psf1.fits',
        'psf2.fits'].
    energies : list of float, Array
        List of energies in keV for which the PSF will be computed.
    pointing_center : tuple or list of float, Array
        List of pointing centers, where each center is a list of coordinates
        [x, y].
    domain : pytree
        The domain over which the PSF will be defined. This typically includes
        the spatial grid or area of interest.
    npatch : int
        Number of patches in the PSF. This divides the domain into smaller
        regions
        for convolution.
    margfrac : float
        Fractional size of the margin, defined as margin/input size. This margin
        is needed to break periodic boundary conditions (PBC) in the patch
        convolution.

    Returns:
    --------
    psf_op : callable
        A function that applies the PSF operator to an input sky array.
    """
    psfs = [_build_tm_erosita_psf_array(psf_file, energies, pcenter,
                                        domain, npatch)
            for psf_file, pcenter in zip(psf_filenames, pointing_center)]
    psfs = np.array(psfs)

    # FIXME: Check sqrt(npatch) and other related parameters

    shp = (domain.shape[-2], domain.shape[-1])
    margin = max((int(np.ceil(margfrac * ss)) for ss in shp))

    def psf_op(x):
        return vmap(linpatch_convolve, in_axes=(None, None, 0, None, None))(
            x, domain, psfs, npatch, margin
        )

    return psf_op


# TODO: split functionality and config loading by implementing
#  build_erosita_response
def build_erosita_response(exposures, exposure_cut=0, tm_ids=None):
    pass


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

    psf_file_names = [join(file_info['psf_path'],
                           'tm' + f'{key}_' + file_info['psf_base_filename'])
                      for key in tel_info['tm_ids']]

    # Get pointings for different telescope modules in RA/DEC
    obs_instance = ErositaObservation(file_info['input'],
                                      file_info['output'],
                                      file_info['obs_path'])
    center_stats = []
    for tm_id in tel_info['tm_ids']:
        tmp_center_stat = obs_instance.get_pointing_coordinates_stats(tm_id,
                                                                      file_info[
                                                                          'input'])
        tmp_center_stat = [tmp_center_stat['RA'][0], tmp_center_stat['DEC'][0]]
        center_stats.append(tmp_center_stat)
    center_stats = np.array(center_stats)

    # center with respect to TM1
    ref_center = center_stats[0]
    d_centers = center_stats - ref_center
    # Set the Image pointing to the center and associate with TM1 pointing
    image_pointing_center = np.array(tuple([cfg['telescope']['fov'] / 2.] * 2))
    pointing_center = d_centers + image_pointing_center

    # FIXME distances for domain energy shouldn't be hardcoded 1
    domain = Domain(tuple([cfg['grid']['edim']] + [cfg['grid']['sdim']] * 2),
                    tuple([1] + [
                        cfg['telescope']['fov'] / cfg['grid']['sdim']] * 2))

    # get psf/exposure/mask function
    psf_func = build_erosita_psf(psf_file_names, psf_info['energy'],
                                 pointing_center,
                                 domain, psf_info['npatch'],
                                 psf_info['margfrac'])

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
        effective_area = calculate_erosita_effective_area(
            file_info['calibration_path'],
            tel_info['tm_ids'],
            np.array(e_min),
            np.array(e_max),
            caldb_folder_name=caldb_folder_name,
            arf_filename_suffix=arf_filename_suffix)
        exposure_func = lambda x: tmp(x) * effective_area[:, :, np.newaxis,
                                           np.newaxis]
    else:
        exposure_func = tmp

    mask_func = build_callable_from_exposure_file(build_readout_function,
                                                  exposure_filenames,
                                                  threshold=tel_info['exp_cut'],
                                                  keys=tel_info['tm_ids'])

    pixel_area = (cfg['telescope']['fov'] / cfg['grid'][
        'sdim']) ** 2  # density to flux
    # plugin
    response_func = lambda x: mask_func(exposure_func(psf_func(x * pixel_area)))
    response_dict = {'pix_area': pixel_area,
                     'psf': psf_func,
                     'exposure': exposure_func,
                     'mask': mask_func,
                     'R': response_func}
    return response_dict


def load_erosita_response():
    pass  # TODO: implement pickle response
