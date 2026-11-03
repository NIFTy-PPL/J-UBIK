# SPDX-License-Identifier: BSD-2-Clause
# Authors: Matteo Guardiani

# Copyright(C) 2026 Max-Planck-Society

# %

"""Load one calibrated, unresampled Keck/NIRC2 narrow-camera exposure.

``load_nirc2_exposure`` returns the ``TargetData`` and ``DataMetaInformation``
the JWST target response takes, in MJy/sr, with sub-pixel centres from the
distortion solution (`nirc2_distortion`). The science image is in DN summed
over coadds, as NIRC2 writes it. NIRC2 flats preserve surface brightness, so
no pixel-area map is applied: the response's block mean over sub-pixels
already predicts the calibrated value.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from nifty.re import logger

from ...color import Color
from ...wcs import subsample_pixel_centers
from ..jwst.data.jwst_data import DataMetaInformation
from ..jwst.data.loader.target_loader import TargetData
from ..jwst.throughput.weights import Throughput
from .nirc2_distortion import DistortionSolution, nirc2_wcs, select_distortion

__all__ = [
    "DetectorNoise",
    "FitsImage",
    "Nirc2Calibration",
    "Nirc2Exposure",
    "load_nirc2_exposure",
]


@dataclass(frozen=True)
class FitsImage:
    """One image HDU, chosen explicitly since reductions differ in layout.

    Header keywords are looked up in this HDU first, then in the primary HDU.
    """

    path: str | Path
    ext: int | str = 0

    def __str__(self) -> str:
        return f"{self.path}[{self.ext}]"


@dataclass(frozen=True)
class Nirc2Calibration:
    """Count rate to F_nu, supplied by the user.

    Parameters
    ----------
    flux_per_count_rate : u.Quantity
        F_nu of a flat-F_nu source that gives a total count rate of 1 DN/s,
        e.g. ``2e-6 * u.Jy / (u.DN / u.s)``. A flat-F_nu reference is what
        `throughput_weights` assumes.
    atmosphere : str
        The atmospheric assumption the calibration holds under, e.g. "above
        the atmosphere, 0.07 mag/airmass removed" or "at the science
        airmass, no extinction correction". Recorded, not applied.
    """

    flux_per_count_rate: u.Quantity
    atmosphere: str

    def __post_init__(self) -> None:
        value = u.Quantity(self.flux_per_count_rate).to_value(u.Jy / (u.DN / u.s))
        if not (np.isfinite(value) and value > 0):
            raise ValueError(f"flux_per_count_rate {value} Jy/(DN/s): must be > 0")
        if not self.atmosphere.strip():
            raise ValueError("the calibration must state its atmospheric assumption")


@dataclass(frozen=True)
class DetectorNoise:
    """Opt-in σ from the detector: shot noise of source and sky, read noise.

    Parameters
    ----------
    gain : float
        Electrons per DN.
    read_noise : float
        Electrons rms of one correlated double sample.
    n_reads : int
        Reads at each end of the ramp per coadd (MCDS, header ``MULTISAM``);
        the read noise falls as ``1 / sqrt(n_reads)`` (white read noise).
    sky : float | np.ndarray
        Sky level per pixel, DN summed over coadds, that was subtracted from
        the science image (0 if the sky is still in it).
    """

    gain: float
    read_noise: float
    n_reads: int
    sky: float | np.ndarray

    def __post_init__(self) -> None:
        if self.gain <= 0 or self.read_noise < 0 or self.n_reads < 1:
            raise ValueError(f"need gain > 0, read_noise >= 0, n_reads >= 1: {self}")

    def sigma(self, data: np.ndarray, coadds: int) -> np.ndarray:
        """σ in DN of ``data`` (DN summed over ``coadds``)."""
        electrons = self.gain * np.maximum(data + self.sky, 0)
        read = coadds * self.read_noise**2 / self.n_reads
        return np.sqrt(electrons + read) / self.gain


@dataclass
class Nirc2Exposure:
    """The JWST target response's inputs, the band and the diagnostics.

    ``target_data`` is a stack of one exposure. Band weights on a sky grid are
    ``throughput_weights(spectral, throughput)``.
    """

    target_data: TargetData
    meta: DataMetaInformation
    throughput: Throughput
    diagnostics: dict


def load_nirc2_exposure(
    science: FitsImage,
    *,
    calibration: Nirc2Calibration,
    throughput: Throughput,
    psf: np.ndarray,
    sigma: FitsImage | None = None,
    detector_noise: DetectorNoise | None = None,
    bad_pixels: FitsImage | None = None,
    pre_2015: DistortionSolution | None = None,
    post_2015: DistortionSolution | None = None,
    distortion: DistortionSolution | None = None,
    subsample: int = 1,
    bunit: str | None = None,
    itime: float | None = None,
    coadds: int | None = None,
    crval: SkyCoord | None = None,
    crpix: tuple[float, float] | None = None,
    position_angle: u.Quantity | None = None,
) -> Nirc2Exposure:
    """Load one NIRC2 narrow-camera exposure.

    Science and σ go together from DN to a count rate, ``DN / (ITIME *
    COADDS)``, and then to MJy/sr with the calibration and the solution's
    pixel solid angle. A keyword argument set to None is read from the
    science headers (keyword in brackets); a missing keyword raises.

    Parameters
    ----------
    science : FitsImage
        Science image, DN summed over coadds, not resampled.
    calibration : Nirc2Calibration
        Count rate to F_nu.
    throughput : Throughput
        Full system throughput of the band (filter, optics, detector, and
        the atmosphere as ``calibration.atmosphere`` states), not the filter
        transmission alone.
    psf : np.ndarray
        The 2-D PSF kernel assigned to this exposure.
    sigma : FitsImage | None
        σ map in DN, like the science.
    detector_noise : DetectorNoise | None
        Opt-in instead of ``sigma``. Exactly one of the two.
    bad_pixels : FitsImage | None
        Non-zero (or NaN) marks a bad pixel.
    pre_2015, post_2015 : DistortionSolution | None
        Solutions chosen by [DATE-OBS], see `select_distortion`.
    distortion : DistortionSolution | None
        Overrides the date-based choice.
    subsample : int
        Sub-pixels per data pixel along each axis.
    bunit : str | None
        [BUNIT], must be DN (or ADU).
    itime : float | None
        [ITIME], seconds per coadd.
    coadds : int | None
        [COADDS].
    crval : SkyCoord | None
        [CRVAL1, CRVAL2], sky position of the reference pixel.
    crpix : tuple[float, float] | None
        [CRPIX1, CRPIX2], FITS 1-based raw position of the reference pixel.
    position_angle : u.Quantity | None
        True position angle of the y axis, East of North. Default
        [ROTPOSN] - [INSTANGL] - beta of the solution, in position-angle
        rotator mode only.

    Returns
    -------
    Nirc2Exposure

    Raises
    ------
    ValueError
        Neither or both of ``sigma`` and ``detector_noise``; not the narrow
        camera; a unit other than DN; shapes that differ from the science
        image (maps of a subarray readout included); a rotator mode other
        than position angle without ``position_angle``.
    KeyError
        A needed header keyword is missing and was not given.
    """
    if (sigma is None) == (detector_noise is None):
        raise ValueError(
            "give exactly one of sigma (a σ map) and detector_noise (opt-in); "
            "no uniform σ is assumed"
        )
    if int(subsample) != subsample or subsample < 1:
        raise ValueError(f"subsample {subsample} must be a positive integer")

    data, header = _read(science)

    def keyword(key, override):
        if override is not None:
            return override
        if key not in header:
            raise KeyError(f"{science}: no {key} in the headers; pass it explicitly")
        return header[key]

    camera = str(header.get("CAMNAME", "narrow")).strip().lower()
    if camera != "narrow":
        raise ValueError(f"{science}: CAMNAME {camera!r}, only the narrow camera")
    unit = str(keyword("BUNIT", bunit)).strip().upper()
    if unit not in ("DN", "ADU"):
        raise ValueError(f"{science}: unit {unit!r}, expected DN summed over coadds")
    itime = float(keyword("ITIME", itime))
    coadds = int(keyword("COADDS", coadds))
    if itime <= 0 or coadds < 1:
        raise ValueError(f"{science}: ITIME {itime} s and COADDS {coadds}")

    if sigma is not None:
        std = _read(sigma)[0]
        sigma_source = str(sigma)
    else:
        std = detector_noise.sigma(data, coadds)
        d = detector_noise
        sigma_source = (
            f"detector noise: gain {d.gain} e-/DN, read noise {d.read_noise} e-, "
            f"{d.n_reads} reads, mean sky {np.mean(d.sky):.6g} DN"
        )
    bad = (
        np.zeros(data.shape, bool) if bad_pixels is None else _read(bad_pixels)[0] != 0
    )

    if distortion is None:
        date_obs = str(keyword("DATE-OBS", None))
        distortion = select_distortion(date_obs, pre_2015, post_2015)
        chosen_by = f"DATE-OBS {date_obs}"
    else:
        chosen_by = "override"
    for name, shape in (
        ("sigma", std.shape),
        ("bad_pixels", bad.shape),
        ("distortion maps", distortion.dx.shape),
    ):
        if shape != data.shape:
            raise ValueError(f"{name} {shape} differ from the science {data.shape}")

    if position_angle is None:
        mode = str(header.get("ROTMODE", "position angle")).strip().lower()
        if mode != "position angle":
            raise ValueError(
                f"{science}: ROTMODE {mode!r}; ROTPOSN - INSTANGL is the sky "
                "position angle only in position-angle mode, pass position_angle"
            )
        nominal = float(keyword("ROTPOSN", None)) - float(keyword("INSTANGL", None))
        position_angle = nominal * u.deg - distortion.angle_offset
    if crval is None:
        crval = SkyCoord(keyword("CRVAL1", None), keyword("CRVAL2", None), unit="deg")
    if crpix is None:
        crpix = (float(keyword("CRPIX1", None)), float(keyword("CRPIX2", None)))

    wcs = nirc2_wcs(distortion, crval, crpix, position_angle)
    ny, nx = data.shape
    centers = subsample_pixel_centers((0, ny, 0, nx), wcs, subsample)

    pixel_scale = distortion.pixel_scale.to(u.deg)
    dvol = pixel_scale**2
    rate = calibration.flux_per_count_rate * u.DN / (itime * coadds * u.s)
    per_dn = (rate / dvol).to_value(u.MJy / u.sr)
    data, std = data * per_dn, std * per_dn
    finite = np.isfinite(data) & np.isfinite(std)
    mask = finite & (std > 0) & ~bad

    target_data = TargetData(
        subsample=int(subsample),
        data=data[None],
        mask=mask[None],
        nan_mask=finite[None],
        std=std[None],
        psf=np.asarray(psf, float)[None],
        subsample_centers=[centers],
    )
    meta = DataMetaInformation(
        unit=u.MJy / u.sr,
        dvol=dvol,
        pixel_scale=pixel_scale,
        color=_pivot(throughput),
    )
    diagnostics = dict(
        science=str(science),
        sigma=sigma_source,
        bad_pixels=None if bad_pixels is None else str(bad_pixels),
        filter=header.get("FILTER"),
        bunit=unit,
        itime=itime,
        coadds=coadds,
        mjy_sr_per_dn=per_dn,
        calibration=f"{calibration.flux_per_count_rate} ({calibration.atmosphere})",
        distortion=f"{distortion.name}, chosen by {chosen_by}",
        position_angle=position_angle.to(u.deg),
        masked_pixels=int(mask.size - mask.sum()),
    )
    logger.info(f"NIRC2 exposure: {diagnostics}")
    return Nirc2Exposure(target_data, meta, throughput, diagnostics)


def _read(image: FitsImage) -> tuple[np.ndarray, dict]:
    """The HDU's data as float, and the primary header updated by the HDU's."""
    with fits.open(image.path) as hdul:
        hdu = hdul[image.ext]
        if hdu.data is None or hdu.data.ndim != 2:
            raise ValueError(f"{image}: not a 2-D image")
        return np.array(hdu.data, float), {**hdul[0].header, **hdu.header}


def _pivot(curve: Throughput) -> Color:
    """Pivot wavelength, ``sqrt(int T lam dlam / int T / lam dlam)``."""
    samples = curve.wavelength.to_value(u.um)
    # a fine grid, so a coarsely sampled curve (a two-point tophat) is exact enough
    lam = np.linspace(samples[0], samples[-1], 100_001)
    t = np.interp(lam, samples, curve.throughput)
    return Color(
        np.sqrt(np.trapezoid(t * lam, lam) / np.trapezoid(t / lam, lam)) * u.um
    )
