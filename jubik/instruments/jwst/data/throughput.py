# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2026 Max-Planck-Society

# %

"""JWST filter transmission curves for `jubik.sky_filter`.

NIRCam mean system throughputs ship with the package (nircam_throughputs_v5.npz);
filters without a packaged curve fall back to a half-power tophat from
JWST_FILTERS. Repack a new STScI release with

    python -m jubik.instruments.jwst.data.throughput --update <mean_throughputs_dir> --version <tag>
"""

import argparse
from functools import cache
from importlib.resources import files
from pathlib import Path

import numpy as np
from nifty.re import logger

from ....sky_filter import Transmission
from .jwst_information import JWST_FILTERS

# Rename together with a new throughput release.
_NPZ = "nircam_throughputs_v5.npz"
_SUFFIX = "_mean_system_throughput.txt"


@cache
def _curves() -> dict[str, np.ndarray]:
    """All arrays of the packaged npz, loaded once."""
    with (
        (files("jubik.instruments.jwst.data") / _NPZ).open("rb") as f,
        np.load(f) as npz,
    ):
        return {k: npz[k] for k in npz.files}


def __getattr__(name: str) -> str:
    # THROUGHPUT_VERSION is read lazily so importing never touches the npz.
    if name == "THROUGHPUT_VERSION":
        return str(_curves()["version"])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def jwst_transmission(filter_name: str) -> Transmission:
    """Transmission curve of a JWST filter.

    Parameters
    ----------
    filter_name : str
        JWST filter name, case-insensitive, e.g. "F444W".

    Returns
    -------
    Transmission
        The packaged throughput curve, or a half-power tophat from
        JWST_FILTERS when no curve is packaged.

    Raises
    ------
    KeyError
        The filter has neither a packaged curve nor a JWST_FILTERS entry.
    """
    name = filter_name.upper()
    curves = _curves()
    if f"{name}_lam_um" in curves:
        return Transmission(curves[f"{name}_lam_um"], curves[f"{name}_T"])
    if name not in JWST_FILTERS:
        raise KeyError(f"{name}: no throughput curve packaged and not in JWST_FILTERS")
    _, _, _, blue, red = JWST_FILTERS[name]
    logger.warning(f"{name}: no throughput curve packaged, half-power tophat")
    return Transmission(np.array([blue, red]), np.ones(2))


def pack_throughputs(source: Path, version: str, out: Path) -> int:
    """Pack STScI mean system throughput files into one npz.

    Parameters
    ----------
    source : Path
        Directory of ``<filter>_mean_system_throughput.txt`` files with the
        header line "Microns Throughput".
    version : str
        Release tag stored under the key "version".
    out : Path
        The npz to write.

    Returns
    -------
    int
        Number of packed curves.
    """
    arrays = {"version": np.array(version)}
    for path in sorted(Path(source).glob(f"*{_SUFFIX}")):
        name = path.name.removesuffix(_SUFFIX).upper()
        lam, T = np.loadtxt(path, skiprows=1, unpack=True)
        arrays[f"{name}_lam_um"] = lam.astype(np.float64)
        arrays[f"{name}_T"] = T.astype(np.float32)
    n_curves = (len(arrays) - 1) // 2
    if n_curves == 0:
        raise FileNotFoundError(f"no *{_SUFFIX} files in {source}")
    np.savez_compressed(out, **arrays)
    return n_curves


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
    out = Path(__file__).with_name(_NPZ)
    n = pack_throughputs(args.update, args.version, out)
    print(f"wrote {n} curves ({args.version}) to {out}")
