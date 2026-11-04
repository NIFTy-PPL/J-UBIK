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
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from nifty.re import logger
from numpy.typing import ArrayLike, NDArray

from ...color import Color
from ...grid import Grid
from ...sky_filter import FilterWeights
from ...wcs import subsample_pixel_centers
from ...wcs.wcs_base import WcsMixin
from ..jwst.data.jwst_data import DataMetaInformation
from ..jwst.data.loader.target_loader import TargetData
from ..jwst.likelihood.likelihood import ShotNoise
from ..jwst.masking import get_mask_from_index_centers_within_rgrid
from ..jwst.throughput.weights import Throughput, throughput_weights

__all__ = [
    "HstDiagnostics",
    "HstExposure",
    "load_hst_exposure",
    "ramp_poisson_factor",
]

FLAM = u.erg / u.s / u.cm**2 / u.AA
# calnica >= 4.2 read weights |i - (n-1)/2|^power by S/N (Fixsen et al. 2000)
FIXSEN_SNR = (5, 10, 20, 50, 100)
FIXSEN_POWERS = (0.0, 0.4, 1.0, 3.0, 6.0, 10.0)
# BUNIT -> whether the file's unit is counts rather than counts/s
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
    the chip, ``n_masked`` the number of cutout pixels with ``mask`` False.
    """

    file: str
    extensions: tuple[tuple[str, int], ...]
    bounds: tuple[int, int, int, int]
    bunit: str
    phot: dict
    exptime: float | None
    dq_bits: int
    distortion: dict
    n_masked: int
    noise: str | None = None


@dataclass(frozen=True)
class HstExposure:
    """`load_hst_exposure` products.

    ``target_data.psf`` is None: a kernel for this exposure needs the detector
    position (``diagnostics.bounds``), so the caller attaches it. ``noise`` is
    the shot-noise model of a NICMOS MULTIACCUM exposure, else None.
    """

    target_data: TargetData
    meta: DataMetaInformation
    weights: FilterWeights
    diagnostics: HstDiagnostics
    noise: ShotNoise | None = None


def load_hst_exposure(
    path: str | Path,
    grid: Grid,
    throughput: Throughput,
    *,
    margin: int,
    dq_bits: int,
    chip: int | None = None,
    exclude: NDArray[np.bool_] | None = None,
    subsample: int = 1,
    sample_times: ArrayLike | None = None,
    flat: NDArray[np.floating] | None = None,
    dark: NDArray[np.floating] | None = None,
) -> HstExposure:
    """Cut one chip of a NICMOS ``_cal`` or WFPC2 ``_flt`` file around the sky grid.

    Data and std (the ERR extension) are in counts/s, or in counts for a WFPC2
    file with BUNIT counts. NICMOS counts are divided by the TIME extension,
    as calnica's UNITCORR divides SCI and ERR by TIME (NICMOS Data Handbook
    v8), so they are rates on every pixel. ``meta.unit`` is the flux density
    of one data unit, PHOTFNU (or PHOTFLAM PHOTPLAM^2 / c) Jy per count/s,
    divided by EXPTIME for WFPC2 counts, and ``meta.dvol`` the pixel area of
    the linear WCS at the reference pixel. The response's unit conversion and
    sum integration then predict the data. HST flats preserve surface
    brightness, so this nominal area is the right one and no pixel-area map is
    applied. The file is opened read-only.

    Parameters
    ----------
    path : str | Path
        The FITS file.
    grid : Grid
        The sky grid. The cutout is the box around its corners plus
        ``margin``; pixels whose centres lie outside it are masked.
    throughput : Throughput
        System throughput curve of the exposure's camera and filter.
    margin : int
        Pixels added on every side of the cutout, at least the PSF half-width.
        The cutout is clipped to the chip.
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
    sample_times : ArrayLike | None
        NICMOS: the MULTIACCUM read times in seconds after the zeroth read
        (whose 0 may be left out): the SAMPTIME keywords of the ``_ima``
        file's SCI extensions. The last must equal EXPTIME; the Instrument
        Handbook's planned times drift from the real ones. None uses the Data
        Handbook's two-read term.
    flat : NDArray[np.floating] | None
        NICMOS: chip-shaped inverse flat that FLATCORR multiplied the data by
        (the TDFFILE imset TDFGROUP, or the FLATFILE). None assumes 1.
    dark : NDArray[np.floating] | None
        NICMOS: chip-shaped dark current and amplifier glow that DARKCORR
        subtracted, as it appears in SCI: count rate times the inverse flat
        and ZPSCALE (times TIME for BUNIT counts, divided by TIME with SCI),
        not the cumulative DARKFILE reads. Its shot noise remains. None is 0.

    Returns
    -------
    HstExposure
        ``target_data`` with a leading exposure axis of length one; ``mask``
        is True where the data enter the likelihood, ``nan_mask`` where data
        and std are finite and std is positive. ``meta`` with PHOTPLAM as
        ``color``. ``weights`` from `throughput_weights` on ``grid.spectral``.
        NICMOS MULTIACCUM ``noise``: ``a`` = ERR², read noise only from
        calnica 4.2 on (NICMOS Data Handbook v8, p. 47), ``b`` the shot noise
        of the least-squares ramp slope, see `_nicmos_noise`. None for other
        modes, whose ERR already holds the Poisson term, and for WFPC2.
        NICMOS pixels with TIME <= 0 are masked in every mode.

    Raises
    ------
    ValueError
        Not a NICMOS or WFPC2 file, ``chip`` does not select one SCI extension,
        the ERR, DQ or NICMOS TIME extension (MULTIACCUM: SAMP, ADCGAIN) is
        missing, a chip map has the wrong shape or is given for another mode,
        the header has no distortion solution, BUNIT or the photometric
        keywords are unusable, the grid misses the chip, or every pixel is
        masked.
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
        unit = u.def_unit(f"{Path(path).stem}_sci{sci.ver}", _jy_per_unit(card, path))
        pixel_area = proj_plane_pixel_area(wcs) * u.deg**2

        bounds = _cutout_bounds(wcs, grid, margin, sci.data.shape, path)
        cut = np.s_[bounds[0] : bounds[1], bounds[2] : bounds[3]]
        data = np.asarray(sci.data[cut], float)
        std = np.asarray(_pixels(err, sci.data.shape)[cut], float)
        nicmos = hdul[0].header["INSTRUME"].strip() == "NICMOS"
        seconds = 1.0
        if nicmos:
            time = _pixels(_extension(hdul, path, "TIME", sci.ver), sci.data.shape)
            time = np.asarray(time[cut], float)
            if BUNIT_IS_COUNTS[str(card("BUNIT")).strip().upper()]:
                seconds = np.where(time > 0, time, np.nan)  # as UNITCORR does
                data, std = data / seconds, std / seconds
        nan_mask = np.isfinite(data) & np.isfinite(std) & (std > 0)
        dq_cut = np.asarray(_pixels(dq, sci.data.shape)[cut], np.int64)
        mask = nan_mask & ((dq_cut & dq_bits) == 0)
        mask &= get_mask_from_index_centers_within_rgrid(bounds, wcs, grid.spatial)
        chip_maps = dict(exclude=exclude, flat=flat, dark=dark)
        for name, value in chip_maps.items():
            if value is not None and np.shape(value) != sci.data.shape:
                raise ValueError(
                    f"{name} has shape {np.shape(value)}, the chip {sci.data.shape}"
                )
            chip_maps[name] = None if value is None else np.asarray(value)[cut]
        if exclude is not None:
            mask &= ~chip_maps["exclude"].astype(bool)
        if chip_maps["dark"] is not None:
            chip_maps["dark"] = chip_maps["dark"] / seconds
        if nicmos:
            mask &= time > 0  # no valid sample

        noise = noise_info = None
        # ACCUM and BRIGHTOBJ ERR already hold the Poisson term (NOISCALC)
        if nicmos and str(card("OBSMODE")).strip() == "MULTIACCUM":
            samp = _extension(hdul, path, "SAMP", sci.ver)
            samp = _pixels(samp, sci.data.shape)[cut]
            noise, noise_info = _nicmos_noise(
                card,
                path,
                data,
                std,
                samp,
                time,
                sample_times,
                chip_maps["flat"],
                chip_maps["dark"],
            )
            # a bad flat or dark pixel
            mask &= np.isfinite(noise.b) & np.isfinite(noise.sky) & (noise.b >= 0)
        elif any(v is not None for v in (sample_times, flat, dark)):
            raise ValueError(
                f"{path}: sample_times, flat and dark apply to NICMOS MULTIACCUM only"
            )
        if not mask.any():
            raise ValueError(
                f"{path}: every pixel of the cutout {bounds} is masked, "
                f"{(~nan_mask).sum()} of them by non-finite data or std <= 0"
            )

        centers = subsample_pixel_centers(bounds, wcs, subsample)
        diagnostics = HstDiagnostics(
            file=path,
            extensions=tuple((h.name, h.ver) for h in (sci, err, dq)),
            bounds=bounds,
            bunit=card("BUNIT"),
            phot={k: card(k) for k in PHOT_KEYS if card(k) is not None},
            exptime=card("EXPTIME"),
            dq_bits=dq_bits,
            distortion={k: card(k) for k in DISTORTION_KEYS if card(k) is not None},
            n_masked=int((~mask).sum()),
            noise=noise_info,
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
            unit=unit,
            dvol=pixel_area,
            pixel_scale=np.sqrt(pixel_area),
            color=color,
        ),
        weights=throughput_weights(grid.spectral, throughput, name=Path(path).name),
        diagnostics=diagnostics,
        noise=noise,
    )


def ramp_poisson_factor(times: ArrayLike, power: float = 0.0) -> float:
    """Poisson variance of the least-squares slope through reads at ``times``
    per unit count rate, wᵀMw (1/s).

    w are the slope's weights when read i of n is weighted |i - (n-1)/2|^power
    (Fixsen et al. 2000, ApJ 545, 1053; power 0 is unweighted), M_jk =
    min(t_j, t_k) the covariance of the accumulated counts at unit rate. For n
    equally spaced reads over T and power 0 it is 6(n²+1) / (5n(n+1)T).
    """
    w = _slope_weights(times, power)
    return float(w @ np.minimum.outer(times, times) @ w)


def _slope_weights(times: ArrayLike, power: float) -> np.ndarray:
    t = np.asarray(times, float)
    v = np.abs(np.arange(t.size) - (t.size - 1) / 2) ** power
    x = np.stack([np.ones_like(t), t])
    return np.linalg.solve((x * v) @ x.T, x * v)[1]


def _nicmos_noise(card, path, data, err, samp, time, sample_times, flat, dark):
    """σ² = ERR² + b · max(SCI + dark, 0) of a NICMOS ``_cal`` cutout.

    SCI is the slope times the inverse flat and ZPSCALE (NICMOS Data Handbook
    v8, FLATCORR and PHOTCALC, pp. 51-52), so b = wᵀMw · flat · ZPSCALE /
    ADCGAIN; SCI and ERR are rates (BUNIT counts divided by TIME). calnica's
    CRIDCALC fits the reads after the zeroth by the optimally weighted least
    squares of Fixsen et al. (2000), with the power chosen by S/N = (y_n -
    y_1) / σ(y_n): a pixel with SAMP reads used reads 1..SAMP, and TIME =
    t_SAMP - t_1. Both were checked on calnica 4.4.1 files, whose _cal SCI
    that fit through the _ima reads reproduces on every full-ramp pixel. A
    cosmic ray splits the ramp and shortens TIME; such pixels usually match no
    read and keep the Data Handbook's two-read term.
    """
    gain = card("ADCGAIN")
    if not (gain and gain > 0):
        raise ValueError(f"{path}: NICMOS needs ADCGAIN > 0, it is {gain}")
    if sample_times is not None and not (
        np.ndim(sample_times) == 1
        and np.size(sample_times) >= 1
        and np.all(np.isfinite(sample_times))
    ):
        raise ValueError(f"sample_times {sample_times}: need finite times")
    time = np.asarray(time, float)
    inverse_flat = 1.0 if flat is None else np.asarray(flat, float)
    per_count = inverse_flat * (card("ZPSCALE") or 1.0) / gain  # SCI per electron
    # the Data Handbook's term (p. 47): two reads over TIME
    factor = 1 / np.where(time > 0, time, np.nan)
    placed, weighted, sampled = 0, 0, int((time > 0).sum())
    if sample_times is not None:
        t = np.union1d(0.0, sample_times)  # the zeroth read, given or not
        # EXPTIME is the last read's SAMPTIME (Data Handbook v8, p. 90)
        if not abs(t[-1] - card("EXPTIME")) < 0.01:
            raise ValueError(
                f"{path}: the last sample time {t[-1]} s is not EXPTIME "
                f"{card('EXPTIME')} s; pass the _ima SAMPTIME values"
            )
        reads = t[1:] - t[1]  # after the zeroth, relative to read 1
        # factors and read-noise norms of the first n reads; a slope needs two
        table = np.full((len(FIXSEN_POWERS), reads.size + 1), np.nan)
        norm = np.full(reads.size + 1, np.nan)
        for k in range(2, reads.size + 1):
            table[:, k] = [ramp_poisson_factor(reads[:k], p) for p in FIXSEN_POWERS]
            norm[k] = np.linalg.norm(_slope_weights(reads[:k], 0.0))
        n = np.clip(np.asarray(samp, int), 0, reads.size)
        # shortcut: most cosmic-ray pixels keep the two-read term; their segment
        # combination is not reproduced, upgrade with the _ima reads and DQ
        ok = (n >= 2) & (n == samp) & (np.abs(reads[n - 1] - time) < 0.01)
        # shortcut: calnica's S/N uses the _ima reads; it is estimated from
        # SCI, the read noise ERR / |w_0| and the flat, which picks calnica's
        # power on 87-99 % of pixels; upgrade by passing the _ima reads
        with np.errstate(invalid="ignore", divide="ignore"):  # ERR 0, bad flat
            signal, read = data * time, err / norm[n]
            snr = signal / np.sqrt(read**2 + np.maximum(signal, 0) * per_count)
        power = np.searchsorted(FIXSEN_SNR, np.nan_to_num(snr), side="right")
        factor = np.where(ok, table[power, n], factor)
        placed, weighted = int(ok.sum()), int((ok & (power > 0)).sum())
    b = factor * per_count
    sky = 0.0 if dark is None else np.asarray(dark, float)
    info = (
        f"ADCGAIN {gain}, ramp from sample times on {placed} of {sampled} "
        f"pixels with samples (the rest, e.g. cosmic-ray segments, from the "
        f"two-read term), {weighted} of them weighted by S/N, flat "
        f"{'given' if flat is not None else 'assumed 1'}, "
        f"dark {'given' if np.ndim(sky) else 'assumed 0'}"
    )
    if placed < sampled or flat is None:
        logger.warning(f"{path}: NICMOS shot noise with {info}")
    return ShotNoise(a=err**2, b=b, sky=sky), info


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


def _pixels(hdu: fits.ImageHDU, shape: tuple[int, int]) -> np.ndarray:
    """The extension's array; HST stores a constant one as a PIXVALUE card."""
    return np.full(shape, hdu.header["PIXVALUE"]) if hdu.data is None else hdu.data


def _jy_per_unit(card, path: str) -> u.Quantity:
    """Flux density of one data unit."""
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
    # NICMOS counts are divided by TIME at load
    if BUNIT_IS_COUNTS[bunit] and str(card("INSTRUME")).strip() != "NICMOS":
        if not (exptime and exptime > 0):
            raise ValueError(
                f"{path}: BUNIT {bunit} needs EXPTIME > 0, it is {exptime}"
            )
        fnu = fnu / exptime
    if not (np.isfinite(fnu.value) and fnu.value > 0):
        raise ValueError(f"{path}: one data unit is {fnu}")
    return fnu


def _cutout_bounds(
    wcs: _ChipWcs,
    grid: Grid,
    margin: int,
    shape: tuple[int, int],
    path: str,
) -> tuple[int, int, int, int]:
    min_row, max_row, min_column, max_column = wcs.bounding_indices_from_world_extrema(
        grid.spatial.world_corners()
    )
    if min_row >= shape[0] or max_row <= 0 or min_column >= shape[1] or max_column <= 0:
        raise ValueError(f"{path}: the sky grid misses the chip")
    return (
        max(min_row - margin, 0),
        min(max_row + margin, shape[0]),
        max(min_column - margin, 0),
        min(max_column + margin, shape[1]),
    )
