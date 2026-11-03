# SPDX-License-Identifier: BSD-2-Clause
# Authors: Matteo Guardiani

# Copyright(C) 2026 Max-Planck-Society

# %

"""One chip of a calibrated, unresampled HST exposure as imaging-response input.

``load_hst_exposure`` reads a NICMOS ``_cal.fits`` (NIC1, NIC2) or a WFPC2
``_flt.fits`` file and returns the ``TargetData`` and ``DataMetaInformation``
that the JWST target response (``build_target_response``) takes, the band
weights of the supplied throughput curve and diagnostics. GEIS and split WFPC2
files are converted upstream.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from numpy.typing import NDArray

from ...color import Color
from ...sky_filter import FilterWeights
from ...wcs import subsample_pixel_centers
from ...wcs.wcs_base import WcsMixin
from ..jwst.data.jwst_data import DataMetaInformation
from ..jwst.data.loader.target_loader import TargetData
from ..jwst.throughput.weights import Throughput, throughput_weights

__all__ = ["HstDiagnostics", "HstExposure", "load_hst_exposure"]

MJY_SR = u.MJy / u.sr
FLAM = u.erg / u.s / u.cm**2 / u.AA
# BUNIT -> whether the values still have to be divided by EXPTIME
BUNIT_IS_COUNTS = {"COUNTS": True, "COUNTS/S": False}
PHOT_KEYS = ("PHOTMODE", "PHOTFNU", "PHOTFLAM", "PHOTPLAM", "PHOTZPT")
DISTORTION_KEYS = (
    "WCSNAME",
    "IDCTAB",
    "A_ORDER",
    "B_ORDER",
    "CPDIS1",
    "CPDIS2",
    "D2IMDIS1",
    "D2IMDIS2",
)


class _ChipWcs(WCS, WcsMixin):
    """The astropy WCS of one chip. Its ``pixel_to_world`` is ``all_pix2world``,
    which applies the SIP and lookup-table distortions stored in the file."""


@dataclass(frozen=True)
class HstDiagnostics:
    """What `load_hst_exposure` read and did.

    ``bounds`` is (min_row, max_row, min_column, max_column) of the cutout on
    the chip, ``to_mjy_sr`` the constant data and std were multiplied by,
    ``n_masked`` the number of cutout pixels with ``mask`` False.
    """

    file: str
    extensions: tuple[tuple[str, int], ...]
    bounds: tuple[int, int, int, int]
    bunit: str
    phot: dict
    exptime: float | None
    to_mjy_sr: float
    dq_bits: int
    distortion: dict
    n_masked: int


@dataclass(frozen=True)
class HstExposure:
    """`load_hst_exposure` products.

    ``target_data.psf`` is None: a kernel for this exposure needs the detector
    position (``diagnostics.bounds``), so the caller attaches it.
    """

    target_data: TargetData
    meta: DataMetaInformation
    weights: FilterWeights
    diagnostics: HstDiagnostics


def load_hst_exposure(
    path: str | Path,
    corners: SkyCoord,
    throughput: Throughput,
    spectral: Color,
    *,
    margin: int,
    dq_bits: int,
    chip: int | None = None,
    exclude: NDArray[np.bool_] | None = None,
    subsample: int = 1,
) -> HstExposure:
    """Cut one chip of a NICMOS ``_cal`` or WFPC2 ``_flt`` file around the target.

    Data and std (the ERR extension) are converted to MJy/sr with one constant:
    PHOTFNU, or PHOTFLAM PHOTPLAM^2 / c, over the pixel area of the linear WCS
    at the reference pixel, divided by EXPTIME when BUNIT is counts. HST flats
    preserve surface brightness, so the response's block-mean pixel integration
    predicts the calibrated image as it is; no pixel-area map is applied. The
    file is opened read-only.

    Parameters
    ----------
    path : str | Path
        The FITS file.
    corners : SkyCoord
        Sky positions the cutout must contain, e.g. the corners of the sky grid.
    throughput : Throughput
        System throughput curve of the exposure's camera and filter.
    spectral : Color
        Sky channel bounds the band weights are computed on.
    margin : int
        Pixels added on every side of the box around ``corners``, at least the
        PSF half-width. The cutout is clipped to the chip.
    dq_bits : int
        DQ bits that mask a pixel.
    chip : int | None
        WFPC2: the SCI extension's DETECTOR (1 = PC1, 2-4 = WF2-WF4). NICMOS:
        the CAMERA, checked against the file. None if the file has one SCI
        extension.
    exclude : NDArray[np.bool_] | None
        Chip-shaped user mask, True excludes the pixel.
    subsample : int
        Sub-pixels per data pixel along each axis for the sky coordinates.

    Returns
    -------
    HstExposure
        ``target_data`` with a leading exposure axis of length one; ``mask``
        is True where the data enter the likelihood, ``nan_mask`` where data
        and std are finite and std is positive. ``meta`` in MJy/sr with the
        pixel area as ``dvol`` and PHOTPLAM as ``color``. ``weights`` from
        `throughput_weights`.

    Raises
    ------
    ValueError
        Not a NICMOS or WFPC2 file, ``chip`` does not select one SCI extension,
        the ERR or DQ extension is missing, the header has no distortion
        solution, BUNIT or the photometric keywords are unusable, the target
        misses the chip, or every pixel is masked.
    """
    path = str(path)
    with fits.open(path) as hdul:
        sci = _select_sci(hdul, path, chip)
        err, dq = (_extension(hdul, path, name, sci.ver) for name in ("ERR", "DQ"))

        def card(key):  # the chip's value first, then the primary header's
            return sci.header.get(key, hdul[0].header.get(key))

        wcs = _ChipWcs(sci.header, hdul)
        if not wcs.has_distortion:
            raise ValueError(
                f"{path}: SCI,{sci.ver} has no distortion solution (SIP or lookup "
                "tables); update the header with updatewcs from STWCS"
            )
        pixel_area = proj_plane_pixel_area(wcs) * u.deg**2
        to_mjy_sr = _to_mjy_sr(card, pixel_area, path)

        bounds = _cutout_bounds(wcs, corners, margin, sci.data.shape, path)
        cut = np.s_[bounds[0] : bounds[1], bounds[2] : bounds[3]]
        data = np.asarray(sci.data[cut], float) * to_mjy_sr
        std = np.asarray(err.data[cut], float) * to_mjy_sr
        nan_mask = np.isfinite(data) & np.isfinite(std) & (std > 0)
        mask = nan_mask & ((dq.data[cut] & dq_bits) == 0)
        if exclude is not None:
            if np.shape(exclude) != sci.data.shape:
                raise ValueError(
                    f"exclude has shape {np.shape(exclude)}, the chip {sci.data.shape}"
                )
            mask &= ~np.asarray(exclude, bool)[cut]
        if not mask.any():
            raise ValueError(f"{path}: every pixel of the cutout {bounds} is masked")

        centers = subsample_pixel_centers(bounds, wcs, subsample)
        diagnostics = HstDiagnostics(
            file=path,
            extensions=tuple((h.name, h.ver) for h in (sci, err, dq)),
            bounds=bounds,
            bunit=card("BUNIT"),
            phot={k: card(k) for k in PHOT_KEYS if card(k) is not None},
            exptime=card("EXPTIME"),
            to_mjy_sr=to_mjy_sr,
            dq_bits=dq_bits,
            distortion={k: card(k) for k in DISTORTION_KEYS if card(k) is not None},
            n_masked=int((~mask).sum()),
        )
        color = Color(card("PHOTPLAM") * u.AA)

    return HstExposure(
        target_data=TargetData(
            subsample=subsample,
            data=data[None],
            mask=mask[None],
            nan_mask=nan_mask[None],
            std=std[None],
            psf=None,
            subsample_centers=[centers],
        ),
        meta=DataMetaInformation(
            unit=MJY_SR,
            dvol=pixel_area,
            pixel_scale=np.sqrt(pixel_area),
            color=color,
        ),
        weights=throughput_weights(spectral, throughput, name=Path(path).name),
        diagnostics=diagnostics,
    )


def _select_sci(hdul: fits.HDUList, path: str, chip: int | None) -> fits.ImageHDU:
    instrument = hdul[0].header.get("INSTRUME", "").strip()
    if instrument not in ("NICMOS", "WFPC2"):
        raise ValueError(f"{path}: INSTRUME is {instrument!r}, not NICMOS or WFPC2")
    scis = [h for h in hdul if h.name == "SCI"]
    # NICMOS has one camera per file, WFPC2 one SCI extension per chip
    chips = [
        (
            hdul[0].header.get("CAMERA")
            if instrument == "NICMOS"
            else h.header.get("DETECTOR")
        )
        for h in scis
    ]
    if chip is None and len(scis) == 1:
        return scis[0]
    if chip is None or chips.count(chip) != 1:
        raise ValueError(
            f"{path}: chip={chip} does not select one SCI extension, their "
            f"{instrument} chips are {chips}"
        )
    return scis[chips.index(chip)]


def _extension(hdul: fits.HDUList, path: str, name: str, ver: int) -> fits.ImageHDU:
    try:
        return hdul[name, ver]
    except KeyError:
        raise ValueError(f"{path}: no {name} extension with EXTVER {ver}") from None


def _to_mjy_sr(card, pixel_area: u.Quantity, path: str) -> float:
    """MJy/sr per unit of the data."""
    bunit = str(card("BUNIT")).strip().upper()
    if bunit not in BUNIT_IS_COUNTS:
        raise ValueError(
            f"{path}: BUNIT {bunit!r} is not one of {list(BUNIT_IS_COUNTS)}"
        )
    photfnu, photflam, photplam, exptime = (
        card(k) for k in ("PHOTFNU", "PHOTFLAM", "PHOTPLAM", "EXPTIME")
    )
    if not photplam:
        raise ValueError(f"{path}: PHOTPLAM is not set")
    if photfnu:
        fnu = photfnu * u.Jy
    elif photflam:
        # PHOTFLAM PHOTPLAM^2 / c is the flat-F_nu calibration throughput_weights assumes
        fnu = (photflam * FLAM).to(u.Jy, u.spectral_density(photplam * u.AA))
    else:
        raise ValueError(f"{path}: neither PHOTFNU nor PHOTFLAM is set")
    if BUNIT_IS_COUNTS[bunit]:
        if not (exptime and exptime > 0):
            raise ValueError(
                f"{path}: BUNIT {bunit} needs EXPTIME > 0, it is {exptime}"
            )
        fnu = fnu / exptime
    factor = (fnu / pixel_area).to_value(MJY_SR)
    if not (np.isfinite(factor) and factor > 0):
        raise ValueError(f"{path}: conversion to MJy/sr is {factor}")
    return factor


def _cutout_bounds(
    wcs: _ChipWcs,
    corners: SkyCoord,
    margin: int,
    shape: tuple[int, int],
    path: str,
) -> tuple[int, int, int, int]:
    min_row, max_row, min_column, max_column = wcs.bounding_indices_from_world_extrema(
        corners
    )
    if min_row >= shape[0] or max_row <= 0 or min_column >= shape[1] or max_column <= 0:
        raise ValueError(f"{path}: the target region misses the chip")
    return (
        max(min_row - margin, 0),
        min(max_row + margin, shape[0]),
        max(min_column - margin, 0),
        min(max_column + margin, shape[1]),
    )
