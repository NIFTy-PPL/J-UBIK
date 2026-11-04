# SPDX-License-Identifier: BSD-2-Clause
# Authors: Matteo Guardiani

# Copyright(C) 2026 Max-Planck-Society

# %

"""Target likelihoods over single exposures, for HST and Keck imaging.

JWST stacks the exposures of a filter into one array under one band key. HST
and Keck exposures of a filter can differ in camera, pixel scale and shape,
so `build_exposure_likelihoods` gives every exposure its own band key,
``{instrument}_{camera}_{filter}_{exposure}``, its own nuisance parameters
and its own likelihood, through the JWST target response and likelihood.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import nifty.re as jft
import numpy as np
from astropy import units as u

from ...grid import Grid
from ...sky_filter import SkyFilter
from .alignment.filter_alignment import FilterAlignment
from .data.jwst_data import DataMetaInformation
from .data.loader.target_loader import TargetData
from .jwst_likelihoods import TargetLikelihoodProducts
from .jwst_response import TargetResponseInput, build_target_response
from .likelihood.likelihood import ShotNoise
from .likelihood.target_likelihood import build_target_likelihood
from .parse.jwst_response import SkyMetaInformation
from .parse.rotation_and_shift.rotation_and_shift import (
    rotation_and_shift_algorithm_config_factory,
)
from .parse.zero_flux_model import ZeroFluxPriorConfigs
from .plotting.residuals import ResidualPlottingInformation
from .rotation_and_shift.coordinates_correction import ShiftAndRotationCorrection
from .throughput.weights import Throughput, throughput_weights

__all__ = ["Exposure", "build_exposure_likelihoods", "read_throughput"]


@dataclass(frozen=True)
class Exposure:
    """One loaded exposure and its PSF.

    Attributes
    ----------
    instrument, camera, filter, exposure : str
        Make the band key ``{instrument}_{camera}_{filter}_{exposure}``; the
        band weights are computed once per instrument, camera and filter.
    target_data : TargetData
        Loader output, one exposure; its ``psf`` is replaced by ``psf``.
    meta : DataMetaInformation
        Loader output; ``unit`` is the flux density of one native data unit.
    throughput : Throughput
        System throughput of the camera and filter.
    psf : np.ndarray | jft.Model | HstPsfKernel | EmpiricalPsfKernel
        A 2-D kernel on the subsampled data grid, a PSF model called as
        ``psf((sky, params))`` (e.g. `PsfGaussianBroadening`), or a kernel
        object, whose sampling and pixel scale are checked against the data.
    noise : ShotNoise | None
        The loader's shot-noise model; None gives the fixed-σ Gaussian on
        ``target_data.std``.
    """

    instrument: str
    camera: str
    filter: str
    exposure: str
    target_data: TargetData
    meta: DataMetaInformation
    throughput: Throughput
    psf: object
    noise: ShotNoise | None = None

    @property
    def band(self) -> str:
        return f"{self.instrument}_{self.camera}_{self.filter}_{self.exposure}"


def build_exposure_likelihoods(
    exposures: list[Exposure],
    grid: Grid,
    sky_domain: dict[str, jft.ShapeWithDtype],
    telescope_cfg: dict,
    sky_unit: u.Unit = u.MJy / u.sr,
) -> TargetLikelihoodProducts:
    """One target likelihood per exposure, summed in `.likelihood`.

    The likelihood is the shot-noise Gaussian of ``exposure.noise`` when the
    loader gives one, else the fixed-σ Gaussian on ``target_data.std``.

    Parameters
    ----------
    exposures : list[Exposure]
        Each with a distinct band key.
    grid : Grid
        The sky grid.
    sky_domain : dict[str, jft.ShapeWithDtype]
        The single sky key of the sky model.
    telescope_cfg : dict
        The JWST telescope keys: ``rotation_and_shift`` (algorithm and
        ``correction_priors``, keyed by band or ``default``), optional
        ``zero_flux`` and ``calibration`` (``default`` and band keys, one
        factor per exposure).
    sky_unit : u.Unit
        Unit of the sky.
    """
    if not (isinstance(sky_domain, dict) and len(sky_domain) == 1):
        raise ValueError("needs a single sky key: the likelihood has nuisance keys")
    ((sky_key, sky_leaf),) = sky_domain.items()
    bands = [e.band for e in exposures]
    if not bands or len(set(bands)) != len(bands):
        raise ValueError(
            f"need exposures with distinct band keys, got {bands}; set the "
            "exposure id of each"
        )

    weights = {}
    for e in exposures:
        key = (e.instrument, e.camera, e.filter)
        if key not in weights:
            weights[key] = (
                e.throughput,
                throughput_weights(grid.spectral, e.throughput, name=e.band),
            )
        elif not _same_curve(weights[key][0], e.throughput):
            raise ValueError(f"{e.band}: another throughput for {key}")
    sky_filter = SkyFilter(
        grid,
        {e.band: weights[e.instrument, e.camera, e.filter][1] for e in exposures},
        sky_key=sky_key,
        dtype=sky_leaf.dtype,
    )
    if sky_filter.domain[sky_key].shape != sky_leaf.shape:
        raise ValueError(
            f"sky_domain shape {sky_leaf.shape} does not match the grid "
            f"shape {sky_filter.domain[sky_key].shape}"
        )

    sky_meta = SkyMetaInformation(
        grid_extension=None, unit=sky_unit, dvol=grid.spatial.dvol
    )
    rotation_and_shift = telescope_cfg["rotation_and_shift"]
    algorithm = rotation_and_shift_algorithm_config_factory(rotation_and_shift)
    zero_flux = ZeroFluxPriorConfigs.from_yaml_dict(telescope_cfg.get("zero_flux"))
    # same layout as zero_flux: a default and per-band priors
    calibration = ZeroFluxPriorConfigs.from_yaml_dict(telescope_cfg.get("calibration"))

    likelihoods = []
    plotting = ResidualPlottingInformation()
    for e in exposures:
        target_data, psf_model = _with_psf(e)
        correction = ShiftAndRotationCorrection(
            domain_key=e.band,
            correction_prior=FilterAlignment(e.band).load_correction_prior(
                rotation_and_shift["correction_priors"], number_of_observations=1
            ),
            rotation_center=None,  # unused by the shift-only correction
        )
        likelihood = build_target_likelihood(
            filter_name=e.band,
            response=build_target_response(
                TargetResponseInput(
                    filter_name=e.band,
                    grid=grid,
                    sky_filter=sky_filter,
                    target_data=target_data,
                    filter_meta=e.meta,
                    sky_meta=sky_meta,
                    rotation_and_shift_algorithm=algorithm,
                    zero_flux_prior_configs=zero_flux,
                    shift_and_rotation_correction=correction,
                    psf_model=psf_model,
                    calibration_prior=(
                        None
                        if calibration is None
                        else calibration.get_name_setting_or_default(e.band)
                    ),
                )
            ),
            target_data=target_data,
            noise=e.noise,
        )
        plotting.append_information(
            filter=e.band,
            data=target_data.data,
            mask=target_data.mask,
            std=target_data.std,
            builder=likelihood.builder,
            meta=e.meta,
        )
        likelihoods.append(likelihood)

    return TargetLikelihoodProducts(
        likelihoods=likelihoods, plotting=plotting, sky_filter=sky_filter
    )


def read_throughput(path: str | Path, wavelength_unit: str = "um") -> Throughput:
    """A two-column text file, wavelength and system throughput."""
    wavelength, throughput = np.loadtxt(path, unpack=True)
    return Throughput(wavelength * u.Unit(wavelength_unit), throughput)


def _same_curve(a: Throughput, b: Throughput) -> bool:
    return np.array_equal(a.wavelength, b.wavelength) and np.array_equal(
        a.throughput, b.throughput
    )


def _with_psf(e: Exposure) -> tuple[TargetData, jft.Model | None]:
    """The target data with the exposure's kernel, or with its PSF model."""
    if isinstance(e.psf, jft.Model):
        return e.target_data, e.psf
    kernel = e.psf
    if hasattr(kernel, "pixel_scale"):  # HstPsfKernel, EmpiricalPsfKernel
        # Tiny Tim and empirical kernels include the response of one kernel
        # pixel; the response integrates over each subsample pixel again
        sampling = getattr(kernel, "subsample", 1)  # empirical: detector-sampled
        if sampling != e.target_data.subsample:
            raise ValueError(
                f"{e.band}: the kernel has {sampling} pixels per detector pixel "
                f"and includes their pixel response, the data subsample is "
                f"{e.target_data.subsample}; use subsample {sampling}"
            )
        # 5% catches a kernel of another camera, not plate-scale drifts
        detector_scale = kernel.pixel_scale * sampling
        data_scale = e.meta.pixel_scale.to_value(u.arcsec)
        if not np.isclose(detector_scale, data_scale, rtol=0.05):
            raise ValueError(
                f"{e.band}: the kernel is for {detector_scale:.4g} arcsec "
                f"pixels, the data have {data_scale:.4g} arcsec"
            )
        kernel = kernel.kernel
    kernel = np.asarray(kernel, float)
    return replace(e.target_data, psf=kernel.reshape((1,) + kernel.shape[-2:])), None
