# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2026 Max-Planck-Society

# %

"""NIRCam filter weights from the STScI mean system throughput curves.

The curves are not shipped with jubik; like the data they are provided by the
user. Download the STScI release once with

    python -m jubik.instruments.jwst.throughput.nircam --download <throughput_dir>

which fetches ``NIRCAM_THROUGHPUTS_URL`` (release
``nircam_throughputs_4Nov2022_v5``) and unpacks its
``<FILTER>_mean_system_throughput.txt`` files into ``throughput_dir``. The jwst
config names that directory under ``files.throughputs``, and
``nircam_filter_weights(spectral, filter_name, throughput_dir)`` reads the
filter's curve from it and hands it to ``throughput_weights``.
"""

import argparse
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

from ....color import Color
from ....sky_filter import FilterWeights
from ..data.jwst_information import nircam_filters
from .weights import MAX_MISSING, throughput_weights

__all__ = [
    "NIRCAM_THROUGHPUTS_URL",
    "download_nircam_throughputs",
    "nircam_filter_weights",
]

# --------------------------------------------------------------------------- #
# Public: a NIRCam filter's weights and the STScI release download
# --------------------------------------------------------------------------- #

# https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-filters
NIRCAM_THROUGHPUTS_URL = (
    "https://jwst-docs.stsci.edu/files/216457506/216457549/1/1762454236677/"
    "nircam_throughputs_4Nov2022_v5.tar.gz"
)


def nircam_filter_weights(
    spectral: Color,
    filter_name: str,
    throughput_dir: str | Path,
    max_missing: float = MAX_MISSING,
) -> FilterWeights:
    """Weights of a NIRCam filter on the sky channels, from its STScI curve.

    Parameters
    ----------
    spectral : Color
        Sky channel bounds.
    filter_name : str
        NIRCam filter name, case-insensitive, e.g. "F444W".
    throughput_dir : str | Path
        Directory holding ``<FILTER>_mean_system_throughput.txt``, see
        `download_nircam_throughputs`.
    max_missing : float
        See `throughput_weights`.

    Returns
    -------
    FilterWeights
        Photon-weighted integrals of the filter's mean system throughput curve.

    Raises
    ------
    KeyError
        Not a NIRCam filter.
    FileNotFoundError
        The filter's curve is not in ``throughput_dir``.
    """
    name = filter_name.upper()
    if name not in nircam_filters:
        raise KeyError(f"{name}: not a NIRCam filter")
    path = Path(throughput_dir) / f"{name}{_CURVE_SUFFIX}"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: no throughput curve for {name}; download the STScI release "
            f"with python -m jubik.instruments.jwst.throughput.nircam --download "
            f"{throughput_dir}"
        )
    wavelength_um, throughput = _read_curve(path)
    return throughput_weights(spectral, wavelength_um, throughput, max_missing, name)


def download_nircam_throughputs(
    throughput_dir: str | Path, url: str = NIRCAM_THROUGHPUTS_URL
) -> int:
    """Download the STScI release and unpack its mean curves into a directory.

    Parameters
    ----------
    throughput_dir : str | Path
        Destination; created if missing. Receives one
        ``<FILTER>_mean_system_throughput.txt`` per filter.
    url : str
        The release archive, a tar.gz with a ``mean_throughputs`` directory.

    Returns
    -------
    int
        Number of curves unpacked.

    Raises
    ------
    FileNotFoundError
        The archive holds no ``mean_throughputs/*_mean_system_throughput.txt``.
    """
    throughput_dir = Path(throughput_dir)
    throughput_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "nircam_throughputs.tar.gz"
        urllib.request.urlretrieve(url, archive)
        return _unpack_mean_curves(archive, throughput_dir)


# --------------------------------------------------------------------------- #
# Internals: the STScI file format
# --------------------------------------------------------------------------- #

_CURVE_SUFFIX = "_mean_system_throughput.txt"
_MEAN_DIR = "mean_throughputs"


def _read_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Wavelengths (microns) and throughput of one STScI ASCII curve.

    Parameters
    ----------
    path : Path
        File with the header line "Microns Throughput" and two columns.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(wavelength_um, throughput)``.
    """
    wavelength_um, throughput = np.loadtxt(path, skiprows=1, unpack=True)
    return wavelength_um, throughput


def _unpack_mean_curves(archive: Path, throughput_dir: Path) -> int:
    """Copy the ``mean_throughputs`` curves out of the release archive.

    Members are written under their base name only, so nothing in the archive
    decides where files land.

    Parameters
    ----------
    archive : Path
        The tar.gz release.
    throughput_dir : Path
        Existing destination directory.

    Returns
    -------
    int
        Number of curves written.

    Raises
    ------
    FileNotFoundError
        No matching member in the archive.
    """
    n_curves = 0
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if not (
                member.isfile()
                and _MEAN_DIR in parts
                and parts[-1].endswith(_CURVE_SUFFIX)
            ):
                continue
            with (
                tar.extractfile(member) as src,
                open(throughput_dir / parts[-1], "wb") as dst,
            ):
                shutil.copyfileobj(src, dst)
            n_curves += 1
    if n_curves == 0:
        raise FileNotFoundError(f"{archive}: no {_MEAN_DIR}/*{_CURVE_SUFFIX} members")
    return n_curves


# --------------------------------------------------------------------------- #
# Command line: download the STScI release
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the NIRCam mean system throughput curves from STScI."
    )
    parser.add_argument(
        "--download",
        type=Path,
        required=True,
        metavar="THROUGHPUT_DIR",
        help="directory to unpack the *_mean_system_throughput.txt files into",
    )
    parser.add_argument(
        "--url", default=NIRCAM_THROUGHPUTS_URL, help="release archive (tar.gz)"
    )
    args = parser.parse_args()
    n = download_nircam_throughputs(args.download, args.url)
    print(f"wrote {n} curves to {args.download}")
