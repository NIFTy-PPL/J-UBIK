# SPDX-License-Identifier: BSD-2-Clause
# Authors: Matteo Guardiani

# Copyright(C) 2026 Max-Planck-Society

# %

"""Load one calibrated, unresampled Keck/NIRC2 narrow-camera exposure.

``load_nirc2_exposure`` returns the ``TargetData`` and ``DataMetaInformation``
the JWST target response takes, with sub-pixel centres from the distortion
solution (`nirc2_distortion`). Data and σ keep the file's units: DN summed
over coadds, as NIRC2 writes them, or DN/s. The calibration enters through
``DataMetaInformation.unit``, a flux-density unit worth one native data unit,
so the response's unit conversion and sum integration predict native data.
``dvol`` is the solution's nominal pixel solid angle; NIRC2 flats preserve
surface brightness, so no pixel-area map is applied.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from nifty.re import logger
from scipy.integrate import trapezoid

from ...color import Color
from ...wcs import WcsAstropy, subsample_pixel_centers
from ..jwst.data.jwst_data import DataMetaInformation
from ..jwst.data.loader.target_loader import TargetData
from ..jwst.likelihood.likelihood import ShotNoise
from ..jwst.masking.data_mask import get_mask_from_index_centers_within_rgrid
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
    """Opt-in noise model from the detector: shot noise of source and sky,
    read noise, σ² = a + b · max(signal + sky, 0) (`ShotNoise`).

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
        Sky level per pixel, DN summed over coadds (also for a DN/s science
        image) as it appears in the flat-fielded science image, that was
        subtracted from it (0 if the sky is still in it). A scalar or the
        science image's shape.
    flat : float | np.ndarray
        The flat the science image was divided by, normalised as applied. A
        scalar or the science image's shape; pixels where it is 0 or NaN are
        masked.
    """

    gain: float
    read_noise: float
    n_reads: int
    sky: float | np.ndarray
    flat: float | np.ndarray = 1.0

    def __post_init__(self) -> None:
        # negated so that NaN fails as well
        if not (
            0 < self.gain < np.inf
            and 0 < self.read_noise < np.inf
            and self.n_reads >= 1
            and not np.any(np.asarray(self.flat) < 0)
        ):
            raise ValueError(
                "need finite gain > 0, read_noise > 0, n_reads >= 1, flat >= 0"
            )

    def shot_noise(self, shape, coadds: int, dn_per_unit: float) -> ShotNoise:
        """a, b and sky in the science unit, worth ``dn_per_unit`` DN summed
        over ``coadds``.

        SCI = raw / flat, so the raw read and shot variances in DN², RN² ·
        coadds / (n_reads gain²) and flat · (SCI + sky) / gain, are divided by
        flat².
        """
        for name, value in (("sky", self.sky), ("flat", self.flat)):
            if np.shape(value) not in ((), shape):
                raise ValueError(
                    f"{name} {np.shape(value)}: a scalar or the science {shape}"
                )
        flat = np.asarray(self.flat, float)
        read = coadds * self.read_noise**2 / self.n_reads / self.gain**2
        # shortcut: the shot noise of an MCDS difference is taken as over the
        # full ITIME; reads within a group span s shorten it by about s/3, so
        # b is high by s/(3 ITIME), exact for CDS; upgrade with read times
        with np.errstate(divide="ignore"):  # flat 0: infinite σ, masked
            return ShotNoise(
                a=read / (flat * dn_per_unit) ** 2,
                b=1 / (self.gain * flat * dn_per_unit),
                sky=np.asarray(self.sky, float) / dn_per_unit,
            )


@dataclass
class Nirc2Exposure:
    """The JWST target response's inputs, the band and the diagnostics.

    ``target_data`` is a stack of one exposure. Band weights on a sky grid are
    ``throughput_weights(spectral, throughput)``. ``noise`` is the
    `DetectorNoise` model in the science unit, None for a σ map.
    """

    target_data: TargetData
    meta: DataMetaInformation
    throughput: Throughput
    diagnostics: dict
    noise: ShotNoise | None = None


def load_nirc2_exposure(
    science: FitsImage,
    *,
    calibration: Nirc2Calibration,
    throughput: Throughput,
    psf: np.ndarray,
    sky_grid: WcsAstropy,
    margin: int,
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

    Data and σ are the file values. ``meta.unit`` is worth
    ``calibration / (ITIME * COADDS)`` Jy for DN, ``calibration`` Jy for
    DN/s. A keyword argument set to None is read from the science headers
    (keyword in brackets); a missing keyword raises.

    Parameters
    ----------
    science : FitsImage
        Science image, DN summed over coadds or DN/s, not resampled.
    calibration : Nirc2Calibration
        Count rate to F_nu.
    throughput : Throughput
        Full system throughput of the band (filter, optics, detector, and
        the atmosphere as ``calibration.atmosphere`` states), not the filter
        transmission alone.
    psf : np.ndarray
        The 2-D PSF kernel assigned to this exposure, sampled on the sub-pixel
        grid (``pixel_scale / subsample`` per pixel) and centred as
        ``PsfStatic`` expects; a detector-sampled kernel needs subsample 1.
    sky_grid : WcsAstropy
        The reconstruction grid; pixels whose centres fall outside it are
        masked, as in the JWST loader.
    margin : int
        Detector pixels added on every side of the box around ``sky_grid``,
        at least the PSF half-width; everything else is cut off. The cutout
        is clipped to the detector.
    sigma : FitsImage | None
        σ map in the science image's unit.
    detector_noise : DetectorNoise | None
        Opt-in instead of ``sigma``. Exactly one of the two. Its model goes
        to ``noise`` for `ShotNoiseGaussianLikelihoodBuilder`; ``std`` is
        that model evaluated at the data, for the fixed-σ Gaussian only.
    bad_pixels : FitsImage | None
        Non-zero (or NaN) marks a bad pixel.
    pre_2015, post_2015 : DistortionSolution | None
        Solutions chosen by [DATE-OBS], see `select_distortion`.
    distortion : DistortionSolution | None
        Overrides the date-based choice.
    subsample : int
        Sub-pixels per data pixel along each axis.
    bunit : str | None
        [BUNIT], DN (or ADU) summed over coadds, or DN/s (ADU/s).
    itime : float | None
        [ITIME], seconds per coadd.
    coadds : int | None
        [COADDS].
    crval : SkyCoord | None
        [CRVAL1, CRVAL2] in [RADESYS] (default ICRS), sky position of the
        reference pixel.
    crpix : tuple[float, float] | None
        [CRPIX1, CRPIX2], FITS 1-based raw position of the reference pixel.
    position_angle : u.Quantity | None
        True position angle of the y axis, East of North. Default
        [ROTPOSN] - [INSTANGL] - beta of the solution, only when [ROTMODE]
        is position angle.

    Returns
    -------
    Nirc2Exposure

    Raises
    ------
    ValueError
        Neither or both of ``sigma`` and ``detector_noise``; not the narrow
        camera; a unit other than DN or DN/s; shapes that differ from the science
        image (maps of a subarray readout included); a rotator mode other
        than position angle without ``position_angle``; the sky grid misses
        the detector; every pixel masked.
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
    subsample = int(subsample)
    if margin < 0:
        raise ValueError(f"margin {margin} must be >= 0")

    data, header = _read(science)

    def keyword(key, override, param):
        if override is not None:
            return override
        if key not in header:
            raise KeyError(f"{science}: no {key} in the headers; pass {param}=")
        return header[key]

    camera = str(header.get("CAMNAME", "narrow")).strip().lower()
    if camera != "narrow":
        raise ValueError(f"{science}: CAMNAME {camera!r}, only the narrow camera")
    unit = str(keyword("BUNIT", bunit, "bunit")).replace(" ", "").upper()
    if unit not in ("DN", "ADU", "DN/S", "ADU/S"):
        raise ValueError(f"{science}: unit {unit!r}, expected DN or DN/s")
    itime = float(keyword("ITIME", itime, "itime"))
    coadds = float(keyword("COADDS", coadds, "coadds"))
    if not (itime > 0 and coadds.is_integer() and coadds >= 1):
        raise ValueError(f"{science}: ITIME {itime} s and COADDS {coadds}")
    coadds = int(coadds)
    seconds = itime * coadds * u.s
    rate_per_unit = u.DN / u.s if unit.endswith("/S") else u.DN / seconds
    jy_per_unit = (calibration.flux_per_count_rate * rate_per_unit).to(u.Jy)
    dn_per_unit = (rate_per_unit * seconds).to_value(u.DN)

    noise = None
    if sigma is not None:
        std = _read(sigma)[0]
        sigma_source = str(sigma)
    else:
        noise = detector_noise.shot_noise(data.shape, coadds, dn_per_unit)
        with np.errstate(invalid="ignore"):  # flat 0: inf · 0, masked
            std = np.sqrt(noise.a + noise.b * np.maximum(data + noise.sky, 0))
        d = detector_noise
        sigma_source = (
            f"detector noise: gain {d.gain} e-/DN, read noise {d.read_noise} e-, "
            f"{d.n_reads} reads, mean sky {np.mean(d.sky):.6g} DN, "
            f"mean flat {np.nanmean(d.flat):.6g}"
        )
    bad = (
        np.zeros(data.shape, bool) if bad_pixels is None else _read(bad_pixels)[0] != 0
    )

    if distortion is None:
        date_obs = str(keyword("DATE-OBS", None, "distortion"))
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
        mode = str(keyword("ROTMODE", None, "position_angle")).strip().lower()
        if mode != "position angle":
            raise ValueError(
                f"{science}: ROTMODE {mode!r}; ROTPOSN - INSTANGL is the sky "
                "position angle only in position-angle mode, pass position_angle"
            )
        rotposn = float(keyword("ROTPOSN", None, "position_angle"))
        nominal = rotposn - float(keyword("INSTANGL", None, "position_angle"))
        position_angle = nominal * u.deg - distortion.angle_offset
    if crval is None:
        ra, dec = (keyword(k, None, "crval") for k in ("CRVAL1", "CRVAL2"))
        frame = header.get("RADESYS", header.get("RADECSYS", "ICRS"))
        crval = SkyCoord(ra, dec, unit="deg", frame=str(frame).strip().lower())
    if crpix is None:
        crpix = tuple(float(keyword(k, None, "crpix")) for k in ("CRPIX1", "CRPIX2"))

    wcs = nirc2_wcs(distortion, crval, crpix, position_angle)
    # the cutout keeps detector pixel indices, so the distortion maps stay whole
    ny, nx = data.shape
    # corners one sky pixel out: the box then holds every pixel the mask keeps
    # (sky coordinates in (0, n), up to half a sky pixel past the far edge).
    # shortcut: box from the four corners; a distortion bending the grid edge
    # by more than half a sky pixel needs a larger margin
    rows0, rows1, cols0, cols1 = wcs.bounding_indices_from_world_extrema(
        sky_grid.world_corners(extension_value=(1, 1))
    )
    if rows0 >= ny or rows1 <= 0 or cols0 >= nx or cols1 <= 0:
        raise ValueError(f"{science}: the sky grid misses the detector")
    bounds = (
        max(rows0 - margin, 0),
        min(rows1 + margin, ny),
        max(cols0 - margin, 0),
        min(cols1 + margin, nx),
    )
    cut = np.s_[bounds[0] : bounds[1], bounds[2] : bounds[3]]
    data, std, bad = data[cut], std[cut], bad[cut]
    if noise is not None:
        noise = ShotNoise(
            *(v[cut] if np.ndim(v) else v for v in (noise.a, noise.b, noise.sky))
        )
    centers = subsample_pixel_centers(bounds, wcs, subsample)

    pixel_scale = distortion.pixel_scale.to(u.deg)
    finite = np.isfinite(data) & np.isfinite(std)
    mask = finite & (std > 0) & ~bad
    mask &= get_mask_from_index_centers_within_rgrid(bounds, wcs, sky_grid)
    if not mask.any():
        raise ValueError(f"{science}: every pixel is masked")

    target_data = TargetData(
        subsample=subsample,
        data=data[None],
        mask=mask[None],
        nan_mask=finite[None],
        std=std[None],
        psf=np.asarray(psf, float)[None],
        subsample_centers=[centers],
    )
    meta = DataMetaInformation(
        unit=u.def_unit(f"nirc2_{unit.lower().replace('/', '_per_')}", jy_per_unit),
        dvol=pixel_scale**2,
        pixel_scale=pixel_scale,
        color=_pivot(throughput),
    )
    diagnostics = dict(
        science=str(science),
        sigma=sigma_source,
        bad_pixels=None if bad_pixels is None else str(bad_pixels),
        camname=header.get("CAMNAME"),
        filter=header.get("FILTER"),
        bunit=unit,
        itime=itime,
        coadds=coadds,
        jy_per_unit=jy_per_unit,
        calibration=f"{calibration.flux_per_count_rate} ({calibration.atmosphere})",
        distortion=f"{distortion.name}, chosen by {chosen_by}",
        position_angle=position_angle.to(u.deg),
        bounds=bounds,
        masked_pixels=int(mask.size - mask.sum()),
    )
    logger.info(f"NIRC2 exposure: {diagnostics}")
    return Nirc2Exposure(target_data, meta, throughput, diagnostics, noise)


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
    return Color(np.sqrt(trapezoid(t * lam, lam) / trapezoid(t / lam, lam)) * u.um)
