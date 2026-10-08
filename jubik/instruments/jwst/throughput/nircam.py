# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2026 Max-Planck-Society

# %

"""NIRCam filter weights from the packaged STScI throughput curves.

``nircam_filter_weights(spectral, filter_name)`` reads the filter's mean system
throughput curve from the npz shipped with the package and hands it to
``throughput_weights``. The curves are release
``nircam_throughputs_4Nov2022_v5``, packaged as ``nircam_throughputs_v5.npz``
in ``jubik.instruments.jwst.data``; ``THROUGHPUT_VERSION`` reports the release.
Repack a new STScI release with

    python -m jubik.instruments.jwst.throughput.nircam --update <mean_throughputs_dir> --version <tag>
"""

import argparse
from functools import cache
from importlib.resources import files
from pathlib import Path

import numpy as np

from ....color import Color
from ....sky_filter import FilterWeights
from ..data.jwst_information import nircam_filters
from .weights import MAX_MISSING, throughput_weights

__all__ = ["nircam_filter_weights", "pack_throughputs"]

# --------------------------------------------------------------------------- #
# Public: a NIRCam filter's weights and the packaged release
# --------------------------------------------------------------------------- #


def nircam_filter_weights(
    spectral: Color, filter_name: str, max_missing: float = MAX_MISSING
) -> FilterWeights:
    """Weights of a NIRCam filter on the sky channels, from its packaged curve.

    Parameters
    ----------
    spectral : Color
        Sky channel bounds.
    filter_name : str
        NIRCam filter name, case-insensitive, e.g. "F444W".
    max_missing : float
        See `throughput_weights`.

    Returns
    -------
    FilterWeights
        Photon-weighted integrals of the packaged system throughput curve.

    Raises
    ------
    KeyError
        Not a NIRCam filter, or a NIRCam filter without a packaged curve.
    """
    name = filter_name.upper()
    if name not in nircam_filters:
        raise KeyError(f"{name}: not a NIRCam filter")
    packaged = _packaged_curves()
    if f"{name}_lam_um" not in packaged:
        raise KeyError(f"{name}: NIRCam filter without a packaged throughput curve")
    return throughput_weights(
        spectral, packaged[f"{name}_lam_um"], packaged[f"{name}_T"], max_missing, name
    )


def pack_throughputs(source_dir: Path, version: str, out_path: Path) -> int:
    """Pack STScI mean system throughput files into one npz.

    Parameters
    ----------
    source_dir : Path
        Directory of ``<filter>_mean_system_throughput.txt`` files with the
        header line "Microns Throughput".
    version : str
        Release tag stored under the key "version".
    out_path : Path
        The npz to write.

    Returns
    -------
    int
        Number of packed curves.

    Raises
    ------
    FileNotFoundError
        ``source_dir`` holds no ``<filter>_mean_system_throughput.txt`` files.
    """
    arrays = {"version": np.array(version)}
    for path in sorted(Path(source_dir).glob(f"*{_STSCI_FILE_SUFFIX}")):
        name = path.name.removesuffix(_STSCI_FILE_SUFFIX).upper()
        wavelength_um, throughput = np.loadtxt(path, skiprows=1, unpack=True)
        arrays[f"{name}_lam_um"] = wavelength_um.astype(np.float64)
        arrays[f"{name}_T"] = throughput.astype(np.float32)
    n_curves = (len(arrays) - 1) // 2
    if n_curves == 0:
        raise FileNotFoundError(f"no *{_STSCI_FILE_SUFFIX} files in {source_dir}")
    np.savez_compressed(out_path, **arrays)
    return n_curves


# Module-level __getattr__ (PEP 562): serves THROUGHPUT_VERSION lazily, so
# importing never touches the npz.
def __getattr__(name: str) -> str:
    """Release tag of the packaged curves, read on first access."""
    if name == "THROUGHPUT_VERSION":
        return str(_packaged_curves()["version"])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --------------------------------------------------------------------------- #
# Internals: the packaged npz
# --------------------------------------------------------------------------- #

# Rename together with a new throughput release.
_PACKAGED_NPZ = "nircam_throughputs_v5.npz"
_STSCI_FILE_SUFFIX = "_mean_system_throughput.txt"


@cache
def _packaged_curves() -> dict[str, np.ndarray]:
    """All arrays of the packaged npz, loaded once.

    Returns
    -------
    dict[str, np.ndarray]
        ``version`` plus ``<FILTER>_lam_um`` and ``<FILTER>_T`` per curve.
    """
    with (
        (files("jubik.instruments.jwst.data") / _PACKAGED_NPZ).open("rb") as f,
        np.load(f) as npz,
    ):
        return {k: npz[k] for k in npz.files}


# --------------------------------------------------------------------------- #
# Command line: repack a new STScI release
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Repack the NIRCam mean system throughputs shipped with jubik."
    )
    parser.add_argument(
        "--update",
        type=Path,
        required=True,
        metavar="MEAN_THROUGHPUTS_DIR",
        help="directory of STScI *_mean_system_throughput.txt files",
    )
    parser.add_argument(
        "--version",
        required=True,
        help="release tag, e.g. nircam_throughputs_4Nov2022_v5",
    )
    args = parser.parse_args()
    out_path = Path(files("jubik.instruments.jwst.data") / _PACKAGED_NPZ)
    n = pack_throughputs(args.update, args.version, out_path)
    print(f"wrote {n} curves ({args.version}) to {out_path}")
