# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2024 Max-Planck-Society

# %
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from os import makedirs
from os.path import isfile, join

import numpy as np
from astropy.coordinates import SkyCoord
from nifty.re import logger
from numpy.typing import ArrayLike

from ..data.jwst_data import JwstData
from ..data.preloader.preloader import _exposure_start_time
from ..parse.jwst_psf import JwstPsfKernelConfig

# JWST datamodel detector names which STPSF spells differently. STPSF accepts
# the NIRCam names (including NRCALONG/NRCBLONG) as they are.
STPSF_DETECTOR_NAMES = {"MIRIMAGE": "MIRIM"}


@lru_cache
def _wss_opd_file(opd_date: str) -> str:
    """The WSS OPD file closest to `opd_date`, from MAST.

    Cached, so star PSFs of one exposure share one MAST query. Only called by
    `build_webb_psf`, so the test seam below still covers it.
    """
    import stpsf

    return stpsf.mast_wss.get_opd_at_time(opd_date, choice="closest")


# --------------------------------
# Testing seam
# --------------------------------
# `test/instruments/jwst/test_jwst_likelihoods.py` monkeypatches `build_webb_psf` so the
# likelihood smoke test never imports stpsf. Keep the stpsf call confined to this
# function so that one patch is enough.
def build_webb_psf(
    camera: str,
    filter: str,
    center_pixel: tuple[float],
    webbpsf_path: str,
    subsample: int,
    fov_arcsec: float | None = None,
    normalize: str = "last",
    *,
    detector: str,
    opd_date: str | None = None,
):
    """
    Builds a Point Spread Function (PSF) model for the JWST using the specified
    camera and filter.

    This function computes the PSF using the `stpsf` library.
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
        The 0-based full-detector pixel position (x, y) for the PSF
        calculation.
    webbpsf_path : str
        The directory holding the STPSF calibration data. It is passed to
        STPSF via the `STPSF_PATH` environment variable.
    subsample : int
        The oversampling factor for the PSF computation.
        This determines how finely the PSF is sampled.
    fov_arcsec : float | None
        The field of view (FOV) in arcseconds.
        If provided, it specifies the size of the PSF image
        in arcseconds.
    detector : str
        The JWST detector name (`meta.instrument.detector`), e.g. 'NRCBLONG'.
    opd_date : str | None
        UTC ISO date-time, e.g. '2023-01-01T00:00:00'. If provided, the
        telescope OPD is the WSS measurement closest to this date, queried from
        MAST once per date and downloaded to `webbpsf_path` on first use. If
        None, STPSF's default OPD is used.
    normalize : str | None
        The normalization method for the PSF.
        Default is 'last', which normalizes based on the
        last computed value. Other options may be supported by `stpsf`.

    Returns
    -------
    numpy.ndarray
        The computed PSF data as a 2D array.

    Raises
    ------
    ImportError
        If `stpsf` is not installed.
    KeyError
        If the specified `camera` is not supported.

    Notes
    -----
    - Ensure that the `stpsf` library is properly installed and that
      `webbpsf_path` points to the STPSF calibration data directory.
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
    >>>     fov_pixels=256,
    >>>     detector='NRCA1',
    >>> )
    """
    try:
        import stpsf
    except ImportError as ie:
        raise ImportError("stpsf is not installed. Please install it first") from ie
    from os import environ

    # STPSF reads its calibration directory from `STPSF_PATH` when an
    # instrument is initialized, so it has to be set before that.
    environ["STPSF_PATH"] = webbpsf_path

    psf_model_supported = {
        "nircam": stpsf.NIRCam,
        "miri": stpsf.MIRI,
    }

    try:
        psf_model_class = psf_model_supported[camera.lower()]
    except KeyError as ke:
        raise KeyError(
            f"You requested the {camera.lower()} STPSF model. "
            f"Supported STPSF models: {list(psf_model_supported)}"
        ) from ke
    psf_model = psf_model_class()
    if opd_date is not None:
        psf_model.load_wss_opd(_wss_opd_file(opd_date), plot=False)

    psf_model.filter = filter.upper()
    # Select the detector before its position, which STPSF interprets on the
    # selected detector.
    detector = detector.upper()
    psf_model.detector = STPSF_DETECTOR_NAMES.get(detector, detector)
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
    subsample: int,
    target_center: SkyCoord,
    config_parameters: JwstPsfKernelConfig,
) -> ArrayLike:
    """
    Loads or computes the Point Spread Function (PSF) kernel for a specified
    camera and filter.

    This function attempts to load a precomputed PSF kernel from a specified
    library path. The file name contains a hash of every input to the PSF
    computation, including normalization, detector, full-detector position,
    and the STPSF package and calibration data versions.
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
        evaluation `psf_arcsec`, the `normalize` key, and `opd_by_date`. If
        `opd_by_date` is set, the PSF uses the WSS OPD measured closest to the
        exposure start, which is then part of the cache key.

    Returns
    -------
    ArrayLike
        The PSF kernel.
    """

    camera = jwst_data.camera.lower()
    filter = jwst_data.filter.lower()
    detector = jwst_data.dm.meta.instrument.detector.upper()
    # The WCS gives 0-based subarray pixels, STPSF expects 0-based full-detector
    # pixels, and `xstart`/`ystart` are the 1-based subarray origin (as in
    # `jwst.assign_wcs.util.subarray_transform`).
    x, y = jwst_data.wcs.world_to_pixel(target_center)
    subarray = jwst_data.dm.meta.subarray
    center_pixel = (x + subarray.xstart - 1, y + subarray.ystart - 1)
    webbpsf_path = config_parameters.webbpsf_path
    psf_library_path = config_parameters.psf_library_path
    psf_arcsec = config_parameters.psf_arcsec
    normalize = config_parameters.normalize
    opd_date = None
    if config_parameters.opd_by_date:
        try:
            opd_date = _exposure_start_time(jwst_data.dm).isot
        except ValueError as e:
            raise ValueError(
                f"psf `opd_by_date` needs the exposure start time. {e} "
                "Set `opd_by_date: false` to use STPSF's default OPD."
            ) from e

    # Calibration identity: STPSF package version and the version of the
    # calibration data in `webbpsf_path`.
    try:
        stpsf_version = version("stpsf")
    except PackageNotFoundError:
        stpsf_version = None
    try:
        with open(join(webbpsf_path, "version.txt")) as f:
            data_version = f.read().strip()
    except OSError:
        data_version = None

    # The file name hashes every input of the PSF computation. Entries written
    # under older, incomplete names are never matched.
    center_pixel_str = f"{int(10 * center_pixel[0])}p{int(10 * center_pixel[1])}"
    cache_key = (
        camera,
        filter,
        detector,
        center_pixel_str,
        psf_arcsec,
        subsample,
        normalize,
        # shortcut: keys on the exposure date, not the OPD file STPSF picks (that
        # needs a MAST query); a newer WSS measurement closer to a recent
        # exposure is not picked up until the cache entry is deleted.
        opd_date,
        stpsf_version,
        data_version,
    )
    cache_hash = sha256(repr(cache_key).encode()).hexdigest()[:16]
    file_name = "_".join(
        (camera, filter, detector.lower(), center_pixel_str, cache_hash)
    )
    path_to_file = join(psf_library_path, file_name)
    makedirs(psf_library_path, exist_ok=True)

    if isfile(path_to_file + ".npy"):
        logger.info(f"Loading PSF {file_name} from {psf_library_path}")
        return np.load(path_to_file + ".npy")

    psf = build_webb_psf(
        camera,
        filter,
        center_pixel,
        webbpsf_path,
        subsample,
        psf_arcsec,
        normalize,
        detector=detector,
        opd_date=opd_date,
    )

    logger.info(f"Saving PSF {file_name} in {psf_library_path}")

    np.save(path_to_file, psf)
    return psf
