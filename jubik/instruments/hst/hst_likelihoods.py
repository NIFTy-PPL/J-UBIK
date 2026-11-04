# SPDX-License-Identifier: BSD-2-Clause
# Authors: Matteo Guardiani

# Copyright(C) 2026 Max-Planck-Society

# %

from pathlib import Path

import nifty.re as jft
from astropy import units as u

from ...grid import Grid
from ..jwst.exposure_likelihoods import (
    Exposure,
    build_exposure_likelihoods,
    read_throughput,
)
from ..jwst.jwst_likelihoods import TargetLikelihoodProducts
from .hst_data import load_hst_exposure
from .tinytim import load_hst_psf_kernel

__all__ = ["build_hst_likelihoods"]


def build_hst_likelihoods(
    cfg: dict,
    grid: Grid,
    sky_domain: dict[str, jft.ShapeWithDtype],
    sky_unit: u.Unit = u.MJy / u.sr,
    files_key: str = "files",
    telescope_key: str = "telescope",
) -> TargetLikelihoodProducts:
    """Load the HST exposures of the config and build one likelihood each.

    ``cfg[files_key]["exposures"]`` lists one entry per exposure with
    ``path`` (NICMOS ``_cal`` or WFPC2 ``_flt``), ``camera`` (e.g. NIC2),
    ``filter``, ``psf`` (a kernel file, `load_hst_psf_kernel`), ``throughput``
    (a two-column text file, `read_throughput`), optional ``wavelength_unit``
    (default um) and ``exposure`` (default: the file stem); the other keys go
    to `load_hst_exposure` (``margin``, ``dq_bits``, ``chip``,
    ``sample_times``). ``cfg[telescope_key]`` holds ``target: subsample``
    and the keys of `build_exposure_likelihoods`.
    """
    telescope = cfg[telescope_key]
    subsample = telescope["target"]["subsample"]
    exposures = []
    for entry in cfg[files_key]["exposures"]:
        entry = dict(entry)
        path, camera, filter, psf = (
            entry.pop(k) for k in ("path", "camera", "filter", "psf")
        )
        throughput = read_throughput(
            entry.pop("throughput"), entry.pop("wavelength_unit", "um")
        )
        exposure_id = entry.pop("exposure", Path(path).stem)
        loaded = load_hst_exposure(path, grid, throughput, subsample=subsample, **entry)
        exposures.append(
            Exposure(
                instrument="hst",
                camera=camera,
                filter=filter,
                exposure=exposure_id,
                target_data=loaded.target_data,
                meta=loaded.meta,
                throughput=throughput,
                psf=load_hst_psf_kernel(psf),
                noise=loaded.noise,
            )
        )
    return build_exposure_likelihoods(exposures, grid, sky_domain, telescope, sky_unit)
