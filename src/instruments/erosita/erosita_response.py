# SPDX-License-Identifier: BSD-2-Clause
# Authors: Vincent Eberle, Matteo Guardiani, Margret Westerkamp

# Copyright(C) 2024 Max-Planck-Society

# %%

from os.path import basename, join
from pathlib import Path

import numpy as np
from astropy.io import fits
from jax import vmap

from .erosita_observation import ErositaObservation
from .erosita_psf import eROSITA_PSF
from ...convolve import linpatch_convolve
from ...data import Domain
from ...response import build_exposure_function, build_readout_function
from ...utils import get_config


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
    function to the loaded exposures.
    The exposure files should be in a .npy or .fits format.
    The loaded exposures are stored in a NumPy array, which is passed as
    input to the callable function. Additional keyword arguments can be
    passed to the callable function using **kwargs.
    The result of applying the callable function to the loaded exposures
    is returned as output.
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
    exposures = np.array(exposures, dtype="float64")
    return builder(exposures, **kwargs)


def calculate_erosita_effective_area(path_to_caldb, tm_ids, e_min, e_max,
                                     caldb_folder_name='caldb',
                                     arf_filename_suffix='_arf_filter_000101v02.fits',
                                     n_points=500):
    """
    Returns the effective area for the given energy range (in keV) and
    telescope module list (in cm^2).
    The effective area is computed by linearly interpolating the effective
    area contained in the ARF file on the desired energy ranges and taking
    the average within each energy range.

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
    pointing_center : list of list, Array
        List of lists containing RA and Dec coordinates (in degrees)
        of the observations' pointing center.
    domain : jubik0.library.data Domain
        The domain over which the PSF will be defined.
        This contains information about the grid on which the PSF is defined.
    npatch : int
        Number of patches in the PSF.
        This divides the domain into smaller regions for convolution.
    margfrac : float
        Specifies the fraction of the zero-padding with respect to the spatial
        domain shape size. This margin is needed to break periodic boundary
        conditions in the patch convolution.

    Returns:
    --------
    psf_op : callable
        A function that applies the PSF operator to an input sky array
        and convolution kernel.
    """
    psfs = [_build_tm_erosita_psf_array(psf_file, energies, pcenter,
                                        domain, npatch)
            for psf_file, pcenter in zip(psf_filenames, pointing_center)]
    psfs = np.array(psfs)

    shp = (domain.shape[-2], domain.shape[-1])
    margin = max((int(np.ceil(margfrac * ss)) for ss in shp))

    def psf_op(x, kernel):
        return vmap(linpatch_convolve, in_axes=(None, None, 0, None, None))(
            x, domain, kernel, npatch, margin
        )

    return psf_op, psfs


def build_erosita_response(
    s_dim,
    e_dim,
    e_min,
    e_max,
    fov,
    tm_ids,
    exposure_filenames,
    psf_filenames,
    psf_energy,
    pointing_center,
    n_patch,
    margfrac,
    exposure_threshold,
    path_to_caldb: str = None,
    caldb_folder_name: str = 'caldb',
    arf_filename_suffix: str = '_arf_filter_000101v02.fits',
    effective_area_correction: bool = True,
):
    """
    Constructs a response function for the eROSITA X-ray telescope,
    incorporating exposure, point spread function (PSF), and mask data.

    Parameters
    ----------
    s_dim : int
        Spatial dimension (number of pixels) along one axis of the image grid.
    e_dim : int
        Energy dimension (number of energy bins).
    e_min : array-like
        Lower bounds of the energy bins (in keV).
    e_max : array-like
        Upper bounds of the energy bins (in keV).
    fov : float
        Field of view (FOV) of the observation, in arcseconds.
    tm_ids : list of int
        List of telescope module IDs to be used.
    exposure_filenames : list of str
        Filenames of the exposure maps.
    psf_filenames : list of str
        Filenames of the PSF data files.
    psf_energy : Array
        Energy levels at which the PSF is defined.
    pointing_center : list of list, Array
        List of lists containing RA and Dec coordinates (in degrees)
        of the observations' pointing center.
    n_patch : int
        Number of patches used in PSF interpolation.
    margfrac : float
        Fraction of the PSF margin to include.
    exposure_threshold : float
        Threshold below which exposure is considered zero.
    path_to_caldb : str
        Path to the calibration database (CALDB).
    caldb_folder_name : str, optional
        Name of the calibration database folder, by default 'caldb'.
    arf_filename_suffix : str, optional
        Suffix for the ARF (Auxiliary Response File) filenames,
        by default '_arf_filter_000101v02.fits'.
    effective_area_correction : bool, optional
        If True, apply effective area correction using the ARF files,
        by default True.

    Returns
    -------
    response_dict : dict
        Dictionary containing the following keys:
            'pix_area': float
                Pixel area corresponding to the FOV and spatial dimension.
            'psf': callable
                PSF function applied over the defined domain.
            'exposure': callable
                Exposure function incorporating optional effective area
                correction.
            'mask': callable
                Mask function derived from exposure maps.
            'R': callable
                Combined response function including PSF, exposure, and mask.
    """
    if not isinstance(e_min, list):
        raise TypeError("e_min must be a list!")

    if not isinstance(e_max, list):
        raise TypeError("e_max must be a list!")

    if len(e_max) != len(e_max):
        raise ValueError("e_min and e_max must have the same length!")

    pixel_area = (fov / s_dim) ** 2  # density to flux

    tmp = build_callable_from_exposure_file(build_exposure_function,
                                            exposure_filenames,
                                            exposure_cut=exposure_threshold, )

    if effective_area_correction:
        if path_to_caldb is None:
            raise ValueError(
                '`path_to_caldb` is required when `effective_area_correction` '
                'is True.'
            )
        effective_area = calculate_erosita_effective_area(
            path_to_caldb,
            tm_ids,
            np.array(e_min),
            np.array(e_max),
            caldb_folder_name=caldb_folder_name,
            arf_filename_suffix=arf_filename_suffix)

        def exposure_func(x):
            return tmp(x) * effective_area[:, :, np.newaxis, np.newaxis]
    else:
        exposure_func = tmp

    mask_func = build_callable_from_exposure_file(
        build_readout_function,
        exposure_filenames,
        threshold=exposure_threshold,
        keys=tm_ids)

    # TODO: enable energy distances
    domain = Domain(tuple([e_dim] + [s_dim] * 2),
                    tuple([1] + [fov / s_dim] * 2))

    psf_func, kernel = build_erosita_psf(psf_filenames,
                                         psf_energy,
                                         pointing_center,
                                         domain,
                                         n_patch,
                                         margfrac)

    def response_func(x, k):
        return mask_func(exposure_func(psf_func(x * pixel_area, k)))

    response_dict = {'pix_area': pixel_area,
                     'psf': psf_func,
                     'exposure': exposure_func,
                     'mask': mask_func,
                     'kernel': kernel,
                     'R': response_func}
    return response_dict


def build_erosita_response_from_config(config_file_path):
    """
    Builds the eROSITA response using configuration settings from a YAML file.

    This function loads a YAML configuration file and uses its entries to build
    the eROSITA response model.
    The response is built based on the telescope modules' Point Spread Function
    (PSF), exposure files, grid information, and pointing coordinates for
    various telescope modules (TMs).

    Parameters
    ----------
    config_file_path : str
        Path to the YAML configuration file.
        For a description of the required fields in the configuration file,
        see demos/erosita_demo.py.
        The file should contain information about the telescope, PSF settings,
        energy grid, and file paths needed to build the eROSITA response.

    Returns
    -------
    response : dict
        The constructed eROSITA response dictionary, which includes
        exposure maps and PSF models.

    Notes
    -----
    - The function assumes that the PSF and exposure files are named according
    to the telescope module ID and the specified filename suffix.
    - Pointing coordinates for the different telescope modules are adjusted
    relative to TM1.
    - Effective area corrections are applied if specified in the configuration.

    Example
    -------
    Assuming the YAML config file contains the required fields, you can build
    the response like this:

    >>> response = build_erosita_response_from_config("/path/to/config.yaml")

    """
    # load config
    cfg = get_config(config_file_path)
    tel_info = cfg['telescope']
    file_info = cfg['files']
    psf_info = cfg['psf']
    grid_info = cfg['grid']

    # load calibration directory paths
    caldb_path = file_info['calibration_path']
    caldb_dir_name = file_info['caldb_folder_name']

    psf_file_suffix = file_info['psf_filename_suffix']

    # load energies
    e_min = grid_info['energy_bin']['e_min']
    e_max = grid_info['energy_bin']['e_max']

    # lists for exposure and psf files
    exposure_filenames = []
    for tm_id in tel_info['tm_ids']:
        exposure_filename = f'tm{tm_id}_' + file_info['exposure']
        [exposure_filenames.append(join(
            file_info['obs_path'],
            "processed",
            f"{Path(exposure_filename).stem}_emin{e}_emax{E}.fits"))
            for e, E in zip(e_min, e_max)]

    psf_file_names = [get_erosita_psf_filenames(caldb_path,
                                                key,
                                                psf_filename_suffix=psf_file_suffix,
                                                caldb_name=caldb_dir_name, )
                      for key in tel_info['tm_ids']]

    # Get pointings for different telescope modules in RA/DEC
    obs_instance = ErositaObservation(file_info['input'],
                                      file_info['output'],
                                      file_info['obs_path'])
    center_stats = []
    for tm_id in tel_info['tm_ids']:
        tmp_center_stat = obs_instance.get_pointing_coordinates_stats(
            tm_id,
            file_info['input'])
        tmp_center_stat = [tmp_center_stat['RA'][0], tmp_center_stat['DEC'][0]]
        center_stats.append(tmp_center_stat)
    center_stats = np.array(center_stats)

    # center with respect to TM1
    ref_center = center_stats[0]
    d_centers = center_stats - ref_center

    # Set the Image pointing to the center and associate with TM1 pointing
    image_pointing_center = np.array(tuple([cfg['telescope']['fov'] / 2.] * 2))
    pointing_center = d_centers + image_pointing_center

    if tel_info['effective_area_correction']:
        caldb_folder_name = 'caldb'
        arf_filename_suffix = '_arf_filter_000101v02.fits'
        if 'caldb_folder_name' in file_info.keys():
            caldb_folder_name = file_info['caldb_folder_name']
        if 'arf_filename_suffix' in file_info.keys():
            arf_filename_suffix = file_info['arf_filename_suffix']

    return build_erosita_response(
        e_dim=grid_info['edim'],
        s_dim=grid_info['sdim'],
        e_min=e_min,
        e_max=e_max,
        exposure_filenames=exposure_filenames,
        psf_filenames=psf_file_names,
        psf_energy=psf_info['energy'],
        pointing_center=pointing_center,
        fov=tel_info['fov'],
        tm_ids=tel_info['tm_ids'],
        n_patch=psf_info['npatch'],
        margfrac=psf_info['margfrac'],
        exposure_threshold=tel_info['exp_cut'],
        path_to_caldb=file_info['calibration_path'],
        caldb_folder_name=caldb_folder_name,
        arf_filename_suffix=arf_filename_suffix
    )


def load_erosita_response():
    pass  # TODO: implement response pickling


def get_erosita_psf_filenames(
    path_to_caldb: str,
    tm_id: int,
    psf_filename_suffix: str = "_2dpsf_190219v05.fits",
    caldb_name: str = "caldb",
) -> str:
    """
    Constructs the full path to an eROSITA PSF (Point Spread Function) file
    for a specific telescope module (TM).

    The path is built in accordance with the standard directory structure
    for eROSITA calibration database (CALDB) data.

    Parameters
    ----------
    path_to_caldb : str
        The base path to the eROSITA calibration database (CALDB) directory.
    tm_id : int
        The telescope module (TM) ID for which the PSF file is being retrieved.
        Must be a valid integer representing the TM (e.g., 1-7).
    psf_filename_suffix : str, optional
        The suffix for the PSF filename. Default is "_2dpsf_190219v05.fits".
    caldb_name : str, optional
        The name of the CALDB folder within the base directory.
        Default is "caldb".

    Returns
    -------
    str
        The full path to the PSF file for the specified telescope module.

    Example
    -------
    >>> get_erosita_psf_filenames("/path/to/caldb", 1)
    '/path/to/caldb/caldb/data/erosita/tm1/bcf/tm1_2dpsf_190219v05.fits'

    Notes
    -----
    - The function assumes the eROSITA PSF files are stored in the following
    format:
      `/path/to/caldb/{caldb_name}/data/erosita/tm{tm_id}/bcf/tm{tm_id}{psf_filename_suffix}`
    """
    path_to_psf_file = join("data", "erosita", f"tm{tm_id}", "bcf")
    path_to_psf_file = join(path_to_caldb, caldb_name, path_to_psf_file)
    filename = f"tm{tm_id}{psf_filename_suffix}"
    return join(path_to_psf_file, filename)

