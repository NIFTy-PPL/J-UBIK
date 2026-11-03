# SPDX-License-Identifier: BSD-2-Clause
# Authors: Matteo Guardiani

# Copyright(C) 2026 Max-Planck-Society

# %
"""HST NICMOS (NIC1, NIC2) and WFPC2 PSF kernels from Tiny Tim.

Tiny Tim 7.5 (https://github.com/spacetelescope/tinytim) is an optional
external tool. `tinytim_kernel` scripts its interactive `tiny1` and runs
`tiny2` (`tiny3` only distorts ACS and WFC3 PSFs), and caches the kernel as a
FITS file that `load_hst_psf_kernel` reads back, also on machines without
Tiny Tim.
"""

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from os.path import isfile, join
from tempfile import mkdtemp

import numpy as np
from astropy.io import fits
from astropy.time import Time
from nifty.re import logger

# The tiny1 answers below follow the prompts of this version.
TINYTIM_VERSION = "7.5"

# tiny1 prints a few kB. Once it rejects an answer (an unknown filter, a PSF
# diameter above its maximum) it re-prompts forever on the exhausted stdin.
MAX_OUTPUT_CHARS = 1_000_000

# tiny1 camera menu choices before and with the NICMOS cryocooler, and the date
# from which tiny1 uses the later cryocooler aberrations.
NICMOS_CHOICES = {"NIC1": ("9", "19", "2002-05-16"), "NIC2": ("10", "20", "2002-09-29")}
# Installed in servicing mission 3B; earlier dates use the cryogen-era tables.
NICMOS_CRYOCOOLER = "2002-03-01"
DETECTOR_SIZE = {"NIC1": 256, "NIC2": 256, "WFPC2": 800}

INSTALL_MESSAGE = (
    "Tiny Tim not found (TINYTIM={tinytim!r}). Build Tiny Tim {version} from "
    "https://github.com/spacetelescope/tinytim (`make linux` or `make macosx`) "
    "and set the TINYTIM environment variable to that directory, as an "
    "absolute path: Tiny Tim "
    "reads its tables from there, and tiny1 and tiny2 are looked up there, "
    "then on PATH."
)


@dataclass(frozen=True)
class HstPsfKernel:
    """An HST PSF kernel and how it is sampled.

    The kernel has odd shape, sums to 1 over its grid, and the PSF centre (the
    chief ray of the requested detector position) is its middle pixel,
    `(shape - 1) / 2`, the convention of `PsfStatic`. Tiny Tim includes the
    pixel response of one kernel pixel: a detector pixel if `subsample == 1`,
    else a subpixel.

    Attributes
    ----------
    kernel : np.ndarray
        The 2-D kernel, normalized to sum 1.
    pixel_scale : float
        Arcsec per kernel pixel.
    subsample : int
        Kernel pixels per detector pixel along an axis.
    charge_diffusion : bool
        Whether the detector charge diffusion is included. Tiny Tim applies it
        to WFPC2 PSFs at detector sampling only, and never to NICMOS.
    """

    kernel: np.ndarray
    pixel_scale: float
    subsample: int
    charge_diffusion: bool


def load_hst_psf_kernel(path: str) -> HstPsfKernel:
    """Load a kernel FITS file, as written by `tinytim_kernel`.

    The primary HDU holds the kernel, of odd shape and centred on its middle
    pixel, and the keywords PIXSCALE (arcsec per kernel pixel), SUBSAMPL
    (kernel pixels per detector pixel) and CHGDIFF (charge diffusion
    included). The kernel is normalized to sum 1.
    """
    with fits.open(path) as hdul:
        kernel = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header
    missing = [key for key in ("PIXSCALE", "SUBSAMPL", "CHGDIFF") if key not in header]
    if missing:
        raise KeyError(f"{path}: missing PSF keywords {missing}")
    if kernel.ndim != 2 or not all(size % 2 for size in kernel.shape):
        raise ValueError(
            f"{path}: the kernel must be 2-D with odd sizes, centred on its "
            f"middle pixel; got shape {kernel.shape}"
        )
    total = kernel.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"{path}: the kernel sum is {total}")
    return HstPsfKernel(
        kernel / total,
        float(header["PIXSCALE"]),
        int(header["SUBSAMPL"]),
        bool(header["CHGDIFF"]),
    )


def _run(args: list[str], cwd: str | None = None, stdin=subprocess.DEVNULL):
    """Run a Tiny Tim program. Returns its exit code and merged output."""
    with subprocess.Popen(
        args,
        cwd=cwd,
        stdin=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    ) as proc:
        output = proc.stdout.read(MAX_OUTPUT_CHARS)
        if len(output) == MAX_OUTPUT_CHARS:
            proc.kill()
    return proc.returncode, output


def _tail(output: str, lines: int = 20) -> str:
    return "\n".join(output.splitlines()[-lines:])


def _run_stage(stage, args, run_dir, expected, stdin=subprocess.DEVNULL):
    """Run tiny1 or tiny2 in `run_dir`, which is kept if it fails."""
    code, output = _run(args, run_dir, stdin)
    with open(join(run_dir, f"{stage}.log"), "w") as f:
        f.write(output)
    if code != 0 or not isfile(join(run_dir, expected)):
        raise RuntimeError(
            f"Tiny Tim {stage} exited with {code} and wrote "
            f"{'' if isfile(join(run_dir, expected)) else 'no '}{expected}. "
            f"Kept {run_dir} (answers in tiny1.in, output in {stage}.log). "
            f"Last output:\n{_tail(output)}"
        )


def _tiny1_answers(
    camera, chip, x, y, date, filter, spectrum, psf_arcsec, subsample, despace
) -> list[str]:
    """The answers to tiny1's prompts, in the order of Tiny Tim 7.5."""
    if camera not in DETECTOR_SIZE:
        raise ValueError(f"camera must be one of {list(DETECTOR_SIZE)}, got {camera}")
    if not all(0 <= p < DETECTOR_SIZE[camera] for p in (x, y)):
        raise ValueError(
            f"{camera} position ({x}, {y}) is off the "
            f"{DETECTOR_SIZE[camera]}x{DETECTOR_SIZE[camera]} pixel detector"
        )
    if not 1 <= subsample <= 10:
        raise ValueError(f"subsample must be 1-10, got {subsample}")

    if camera == "WFPC2":
        if chip not in (1, 2, 3, 4):
            raise ValueError(f"WFPC2 chip must be 1 (PC) or 2-4 (WF), got {chip}")
        answers = ["6", f"{x} {y}"] if chip == 1 else ["5", str(chip), f"{x} {y}"]
    else:
        if date is None:
            raise ValueError(f"{camera} needs the observation date")
        date = Time(date)
        before, after, later_tables = NICMOS_CHOICES[camera]
        if date < Time(NICMOS_CRYOCOOLER):
            day = date.datetime
            answers = [before, f"{x} {y}", f"{day.day} {day.month} {day.year}"]
        else:
            answers = [after, f"{x} {y}", "Y" if date >= Time(later_tables) else "N"]

    answers.append(filter)
    # tiny1 asks for a spectrum only if the filter has more than one wavelength
    # in its tables; for WFPC2 and NICMOS that is every filter but the F...N.
    if filter.endswith("N"):
        if spectrum is not None:
            raise ValueError(f"{filter} is monochromatic in Tiny Tim: no spectrum")
    else:
        # Flat F_nu is the power law F_nu = nu^0.
        answers += ["3", "0"] if spectrum is None else ["5", "spectrum.dat"]
    answers.append(str(float(psf_arcsec)))
    answers += ["N"] if subsample == 1 else ["Y", str(subsample)]
    answers += [str(float(despace)), "psf"]
    return answers


def tinytim_kernel(
    camera: str,
    filter: str,
    position: tuple[float, float],
    psf_library_path: str,
    *,
    chip: int | None = None,
    date: str | None = None,
    spectrum: str | None = None,
    subsample: int = 1,
    psf_arcsec: float = 3.0,
    despace: float = 0.0,
    jitter: float | None = None,
) -> HstPsfKernel:
    """Load a cached Tiny Tim PSF kernel, or compute and cache it.

    The cache file name hashes everything that determines Tiny Tim's output:
    the tiny1 answers and command line, the spectrum file contents and the
    Tiny Tim version. A cache hit needs no Tiny Tim install.

    Parameters
    ----------
    camera : str
        'NIC1', 'NIC2' or 'WFPC2'.
    filter : str
        The filter as named in Tiny Tim's tables, e.g. 'F160W'. Narrow-band
        filters (F...N) are monochromatic there and take no spectrum.
    position : tuple of float
        Detector pixel (x, y) of the PSF centre, rounded to integers as tiny1
        takes them (0-255 for NICMOS, 0-799 for WFPC2).
    psf_library_path : str
        The cache directory. A failed run is kept there as `tinytim_*`.
    chip : int | None
        WFPC2 only: 1 (PC) or 2-4 (WF).
    date : str | None
        NICMOS only: the observation date, in any `astropy.time.Time` format.
        It selects the aberrations before or with the cryocooler.
    spectrum : str | None
        A Tiny Tim spectrum table (first line FNU, FLAM, JY or PHOTLAM, then
        wavelength in Angstrom and flux). None means flat F_nu.
    subsample : int
        Kernel pixels per detector pixel along an axis, 1-10.
    psf_arcsec : float
        PSF diameter in arcsec; tiny1 rounds it down to an even number of
        detector pixels.
    despace : float
        Focus, as secondary mirror despace in microns.
    jitter : float | None
        RMS Gaussian jitter in mas (0.5-200), or None for none.

    Returns
    -------
    HstPsfKernel
    """
    camera, filter = camera.upper(), filter.upper()
    x, y = (int(round(p)) for p in position)
    answers = _tiny1_answers(
        camera, chip, x, y, date, filter, spectrum, psf_arcsec, subsample, despace
    )
    options = [] if jitter is None else [f"jitter={float(jitter)}"]
    spectrum_bytes = None
    if spectrum is not None:
        with open(spectrum, "rb") as f:
            spectrum_bytes = f.read()

    # Runs with any other version are refused below.
    cache_key = (answers, options, spectrum_bytes, TINYTIM_VERSION)
    cache_hash = sha256(repr(cache_key).encode()).hexdigest()[:16]
    detector = f"wfpc2_{chip}" if camera == "WFPC2" else camera.lower()
    file_name = f"{detector}_{filter.lower()}_{x}x{y}_{cache_hash}.fits"
    path_to_file = join(psf_library_path, file_name)
    os.makedirs(psf_library_path, exist_ok=True)

    if isfile(path_to_file):
        logger.info(f"Loading PSF {file_name} from {psf_library_path}")
        return load_hst_psf_kernel(path_to_file)

    tinytim = os.environ.get("TINYTIM", "")
    search_path = os.pathsep.join((tinytim, os.environ.get("PATH", "")))
    tiny1, tiny2 = (shutil.which(name, path=search_path) for name in ("tiny1", "tiny2"))
    # Tiny Tim reads its tables from TINYTIM, so it must not depend on the
    # directory tiny1 and tiny2 run in.
    if not os.path.isabs(tinytim) or tiny1 is None or tiny2 is None:
        raise FileNotFoundError(
            INSTALL_MESSAGE.format(tinytim=tinytim, version=TINYTIM_VERSION)
        )
    code, output = _run([tiny1, "-v"])
    if code != 0 or output.strip() != TINYTIM_VERSION:
        raise RuntimeError(
            f"{tiny1} -v exited with {code}; the tiny1 answers are scripted for "
            f"Tiny Tim {TINYTIM_VERSION}. Output:\n{_tail(output)}"
        )

    run_dir = mkdtemp(prefix="tinytim_", dir=psf_library_path)
    if spectrum is not None:
        shutil.copyfile(spectrum, join(run_dir, "spectrum.dat"))
    with open(join(run_dir, "tiny1.in"), "w") as f:
        f.write("\n".join(answers) + "\n")

    with open(join(run_dir, "tiny1.in")) as stdin:
        _run_stage(
            "tiny1", [tiny1, "tiny.param", *options], run_dir, "tiny.param", stdin
        )
    _run_stage("tiny2", [tiny2, "tiny.param"], run_dir, "psf00.fits")

    with open(join(run_dir, "tiny.param")) as f:
        scale = re.search(r"^\s*(\S+)\s*#\s*Integrated PSF scaling", f.read(), re.M)
    if scale is None:
        raise RuntimeError(f"No PSF scale in {join(run_dir, 'tiny.param')}")
    psf = fits.getdata(join(run_dir, "psf00.fits")).astype(float)
    # Tiny Tim centres the PSF on pixel n/2 of its even grid. A zero row and
    # column at the end make that the middle pixel of an odd grid.
    kernel = np.pad(psf, ((0, 1), (0, 1)))
    hdu = fits.PrimaryHDU(kernel / kernel.sum())
    hdu.header["PIXSCALE"] = (float(scale.group(1)), "arcsec per kernel pixel")
    hdu.header["SUBSAMPL"] = (subsample, "kernel pixels per detector pixel")
    hdu.header["CHGDIFF"] = (
        camera == "WFPC2" and subsample == 1,
        "charge diffusion included",
    )
    hdu.header["TTVERS"] = (TINYTIM_VERSION, "Tiny Tim version")

    logger.info(f"Saving PSF {file_name} in {psf_library_path}")
    # Written next to the cache entry and renamed, so a reader never sees a
    # partial file.
    hdu.writeto(join(run_dir, "kernel.fits"))
    psf_kernel = load_hst_psf_kernel(join(run_dir, "kernel.fits"))
    os.replace(join(run_dir, "kernel.fits"), path_to_file)
    shutil.rmtree(run_dir)
    return psf_kernel
