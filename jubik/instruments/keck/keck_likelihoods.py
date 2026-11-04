# SPDX-License-Identifier: BSD-2-Clause
# Authors: Matteo Guardiani

# Copyright(C) 2026 Max-Planck-Society

# %

from pathlib import Path

import nifty.re as jft
import numpy as np
from astropy import units as u

from ...grid import Grid
from ..jwst.exposure_likelihoods import (
    Exposure,
    build_exposure_likelihoods,
    read_throughput,
)
from ..jwst.jwst_likelihoods import TargetLikelihoodProducts
from .empirical_psf import kernel_from_fits
from .nirc2_distortion import service2016, yelda2010
from .nirc2_loader import (
    DetectorNoise,
    FitsImage,
    Nirc2Calibration,
    load_nirc2_exposure,
)

__all__ = ["build_keck_likelihoods"]


def build_keck_likelihoods(
    cfg: dict,
    grid: Grid,
    sky_domain: dict[str, jft.ShapeWithDtype],
    sky_unit: u.Unit = u.MJy / u.sr,
    files_key: str = "files",
    telescope_key: str = "telescope",
) -> TargetLikelihoodProducts:
    """Load the NIRC2 exposures of the config and build one likelihood each.

    ``cfg[files_key]["exposures"]`` lists one entry per exposure with
    ``science`` (a path, or [path, extension]), ``filter``, ``psf`` (a
    prepared kernel file, `kernel_from_fits`), ``throughput`` (a two-column
    text file, `read_throughput`), ``flux_per_count_rate`` (Jy per DN/s) and
    ``atmosphere`` (`Nirc2Calibration`), and either ``sigma`` (a σ image) or
    ``detector_noise`` (the `DetectorNoise` fields); optional
    ``wavelength_unit`` (default um), ``exposure`` (default: the file stem),
    ``bad_pixels`` and the header overrides of `load_nirc2_exposure`
    (``bunit``, ``itime``, ``coadds``). ``cfg[telescope_key]`` holds
    ``target: subsample``, ``distortion: {pre_2015: [x_map, y_map],
    post_2015: [x_map, y_map]}`` (either may be left out) and the keys of
    `build_exposure_likelihoods`.
    """
    telescope = cfg[telescope_key]
    subsample = telescope["target"]["subsample"]
    maps = telescope["distortion"]
    solutions = dict(
        pre_2015=yelda2010(*maps["pre_2015"]) if "pre_2015" in maps else None,
        post_2015=service2016(*maps["post_2015"]) if "post_2015" in maps else None,
    )
    exposures = []
    for entry in cfg[files_key]["exposures"]:
        entry = dict(entry)
        science = _fits_image(entry.pop("science"))
        filter, psf = entry.pop("filter"), entry.pop("psf")
        throughput = read_throughput(
            entry.pop("throughput"), entry.pop("wavelength_unit", "um")
        )
        calibration = Nirc2Calibration(
            entry.pop("flux_per_count_rate") * u.Jy / (u.DN / u.s),
            entry.pop("atmosphere"),
        )
        exposure_id = entry.pop("exposure", Path(science.path).stem)
        for key in ("sigma", "bad_pixels"):
            if key in entry:
                entry[key] = _fits_image(entry[key])
        if "detector_noise" in entry:
            entry["detector_noise"] = DetectorNoise(**entry["detector_noise"])
        loaded = load_nirc2_exposure(
            science,
            calibration=calibration,
            throughput=throughput,
            psf=np.ones((1, 1)),  # placeholder, the builder attaches the kernel
            sky_grid=grid.spatial,
            subsample=subsample,
            **solutions,
            **entry,
        )
        kernel = kernel_from_fits(
            psf,
            exposure=exposure_id,
            pixel_scale=loaded.meta.pixel_scale.to_value(u.arcsec),
        )
        exposures.append(
            Exposure(
                instrument="keck",
                camera="nirc2",
                filter=filter,
                exposure=exposure_id,
                target_data=loaded.target_data,
                meta=loaded.meta,
                throughput=throughput,
                psf=kernel,
                noise=loaded.noise,
            )
        )
    return build_exposure_likelihoods(exposures, grid, sky_domain, telescope, sky_unit)


def _fits_image(value: str | list) -> FitsImage:
    return FitsImage(value) if isinstance(value, str) else FitsImage(*value)
