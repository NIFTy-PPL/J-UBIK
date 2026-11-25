# SPDX-License-Identifier: BSD-2-Clause
# Authors: Matteo Guardiani

# Copyright(C) 2026 Max-Planck-Society

"""A static per-pixel pattern fixed to the detector, measured from dithered exposures and subtracted.

The pipeline's dark subtraction stamps the dark reference's own per-pixel noise onto every exposure taken
with that reference (in long-wavelength NIRCam, a reference built from few integrations). Dithers move the
sky across the detector, so where an exposure sees sky the pattern is the per-pixel median of the others'
detector-frame residuals. Only numpy, scipy and astropy are needed: the module runs without the jwst package.
"""

import warnings
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.ndimage import binary_dilation, map_coordinates

DO_NOT_USE = 1
N_AMPLIFIERS = 4  # NIRCam full frame: four amplifiers of 512 columns
N_REFERENCE = 4  # reference pixels on each edge
SAME_PER_PATTERN = ("DETECTOR", "FILTER", "R_DARK")


def _robust_std(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def _nanmedian(stack, axis):
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", RuntimeWarning
        )  # all-NaN slices give NaN, handled by the callers
        return np.nanmedian(stack, axis=axis)


def _smooth_background(image, sky, block):
    """Block medians over `sky`, bilinear between block centres, constant beyond the outer ones."""
    ny, nx = image.shape
    blocks = np.where(sky, image, np.nan).reshape(
        ny // block, block, nx // block, block
    )
    med = _nanmedian(blocks, axis=(1, 3))
    med = np.where(np.isfinite(med), med, np.nanmedian(med))
    cy, cx = ((np.arange(n) + 0.5) / block - 0.5 for n in (ny, nx))
    return map_coordinates(
        med, np.meshgrid(cy, cx, indexing="ij"), order=1, mode="nearest"
    )


def _residual(sci, dq, wcs, target, radius, block):
    """Detector-frame residual on sky pixels, NaN elsewhere.

    Sky: not DO_NOT_USE, not a reference pixel, within 3 robust sigma above (5 below) the median with
    sources grown by 3 px, and outside the target disk. The smooth background and each amplifier's
    per-row median are removed.
    """
    ny, nx = sci.shape
    ok = np.zeros(sci.shape, bool)
    ok[N_REFERENCE:-N_REFERENCE, N_REFERENCE:-N_REFERENCE] = True
    ok &= np.isfinite(sci) & ((dq & DO_NOT_USE) == 0)
    med, s = np.median(sci[ok]), _robust_std(sci[ok])
    sky = (
        ok
        & ~binary_dilation(ok & (sci > med + 3 * s), iterations=3)
        & (sci > med - 5 * s)
    )
    x0, y0 = wcs.world_to_pixel(target)
    r_pix = radius.to_value(u.deg) / np.mean(proj_plane_pixel_scales(wcs.celestial))
    yy, xx = np.ogrid[:ny, :nx]
    sky &= np.hypot(xx - x0, yy - y0) > r_pix

    res = np.where(sky, sci - _smooth_background(sci, sky, block), np.nan)
    res = res.reshape(ny, N_AMPLIFIERS, nx // N_AMPLIFIERS)
    res -= np.nan_to_num(_nanmedian(res, axis=2)[..., None])
    return res.reshape(ny, nx).astype(np.float32)


def sky_residuals(paths, target, radius, block=64):
    """Stack of the exposures' sky residuals, in the order of `paths` (SCI units, NaN off sky).

    Parameters
    ----------
    paths: cal files of one detector, filter and dark reference, full frame.
    target: SkyCoord of the target, masked in every exposure.
    radius: angular radius of the target disk (generous: the target's halo must not reach the pattern).
    block: side of the smooth background's blocks, in pixels.
    """
    heads = [fits.getheader(p) for p in paths]
    for key in SAME_PER_PATTERN:
        values = {h[key] for h in heads}
        if len(values) > 1:
            raise ValueError(f"a pattern belongs to one {key}, got {sorted(values)}")
    if any(h["SUBARRAY"] != "FULL" for h in heads):
        raise ValueError(
            "full-frame exposures only: the stripes are removed per 512-column amplifier"
        )
    stack = []
    for p in paths:
        with fits.open(p) as hdul:
            sci, dq = hdul["SCI"].data, hdul["DQ"].data
            stack.append(
                _residual(sci, dq, WCS(hdul["SCI"].header), target, radius, block)
            )
    return np.array(stack)


def static_pattern(stack, min_count):
    """Per-pixel median of the residual stack, 0 where fewer than `min_count` exposures see sky; and the count."""
    count = np.isfinite(stack).sum(axis=0)
    pattern = np.where(count >= min_count, _nanmedian(stack, axis=0), 0)
    return pattern.astype(stack.dtype), count


def subtract_static_pattern(
    paths, target, radius, pattern_path, min_count=5, suffix="staticpattern", block=64
):
    """Subtract the static pattern from each exposure and write `<stem>_<suffix>.fits` beside it.

    Exposure k gets SCI - P_-k, the pattern from the other exposures, so the correction removes none of
    its own noise. ERR = sqrt(ERR_PIPE**2 + sigma_P**2) per pixel, with sigma_P = sqrt(pi/2) s / sqrt(n_-k)
    the noise of a median of n_-k sky residuals of scatter s. Pixels with n_-k < min_count, DO_NOT_USE or
    NaN keep their SCI and the pattern, so their ERR adds the input's per-exposure term
    sigma_x**2 = max(0, clipped sky std**2 - median ERR_PIPE**2). Without an ERR_PIPE
    extension the input ERR is the pipeline's and is kept as ERR_PIPE. The pattern from all exposures and
    its count go to `pattern_path`. No file is ever overwritten.

    Returns one row per exposure: the sigma_x rule on the input and on the corrected sky (clipped sky std,
    sigma_x) against sqrt(median ERR_PIPE**2) and the median sigma_P on the same pixels.
    """
    paths = [Path(p) for p in paths]
    pattern_path = Path(pattern_path)
    outs = [p.with_name(f"{p.stem}_{suffix}.fits") for p in paths]
    existing = [str(o) for o in [*outs, pattern_path] if o.exists()]
    if existing:
        raise FileExistsError(f"not overwritten: {existing}")

    stack = sky_residuals(paths, target, radius, block)
    pattern, count = static_pattern(stack, min_count)
    used = np.isfinite(stack) & (count >= min_count)
    # shortcut: one scatter for the set (sky noise is nearly uniform); per pixel if the background varies
    scatter = _robust_std((stack - pattern)[used])
    head = fits.getheader(paths[0])
    pattern_head = fits.Header([(k, head[k]) for k in SAME_PER_PATTERN])
    pattern_head["PATNMIN"] = (min_count, "PATTERN is 0 where COUNT < PATNMIN")
    pattern_head["PATSCAT"] = (float(scatter), "residual scatter around PATTERN")
    fits.HDUList(
        [
            fits.PrimaryHDU(header=pattern_head),
            fits.ImageHDU(pattern, name="PATTERN"),
            fits.ImageHDU(count.astype(np.int16), name="COUNT"),
        ]
    ).writeto(pattern_path)

    rows = []
    for k, (path, out) in enumerate(zip(paths, outs)):
        own = stack[k].copy()
        stack[k] = np.nan
        pattern_k, count_k = static_pattern(stack, min_count)
        stack[k] = own
        with fits.open(path) as hdul:
            sci, dq = hdul["SCI"].data, hdul["DQ"].data
            err_pipe = hdul["ERR_PIPE"].data if "ERR_PIPE" in hdul else hdul["ERR"].data
            corrected = (count_k >= min_count) & ((dq & DO_NOT_USE) == 0)
            new = np.where(corrected, sci - pattern_k, sci).astype(sci.dtype)
            sigma_p = np.sqrt(np.pi / 2) * scatter / np.sqrt(np.maximum(count_k, 1))
            # the sigma_x rule on the exposure's own sky pixels: the input's, and the corrected data's
            sky = np.isfinite(own) & (err_pipe > 0)
            err_med = np.sqrt(np.median(err_pipe[sky] ** 2))
            sky_in = sigma_clipped_stats(sci[sky], sigma=3)[2]
            sky_out = sigma_clipped_stats(new[sky & corrected], sigma=3)[2]
            sigma_x = np.sqrt(max(0.0, sky_in**2 - err_med**2))
            sigma = np.where(corrected, sigma_p, sigma_x)
            err = np.sqrt(err_pipe.astype(np.float64) ** 2 + sigma**2).astype(
                err_pipe.dtype
            )

            if "ERR_PIPE" not in hdul:
                pipe = fits.ImageHDU(
                    err_pipe, header=hdul["ERR"].header.copy(), name="ERR_PIPE"
                )
                hdul.append(pipe)
            hdul["SCI"].data, hdul["ERR"].data = new, err
            for h in (hdul[0].header, hdul["ERR"].header):
                h["ERRSIGX"] = (sigma_x, "ERR term where not corrected")
            hdul[0].header["PATFILE"] = (pattern_path.name, "pattern, leave-one-out")
            hdul[0].header["PATNMIN"] = (
                min_count,
                "corrected where >= PATNMIN others see sky",
            )
            hdul[0].header["PATSCAT"] = (
                float(scatter),
                "ERR adds sqrt(pi/2) PATSCAT / sqrt(n)",
            )
            hdul.writeto(out)
        rows.append(
            dict(
                file=out.name,
                err_med=float(err_med),
                sky_std_in=float(sky_in),
                sky_std_out=float(sky_out),
                sigma_x_in=float(sigma_x),
                sigma_x_out=float(np.sqrt(max(0.0, sky_out**2 - err_med**2))),
                sigma_p_med=float(np.median(sigma_p[sky & corrected])),
            )
        )
    return rows
