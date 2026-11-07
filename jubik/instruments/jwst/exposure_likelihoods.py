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
from ...models.distributions import build_parametric_prior_from_prior_config
from ...parse.models.distributions import (
    DeltaPriorConfig,
    ProbabilityConfig,
    UniformPriorConfig,
)
from ...sky_filter import SkyFilter
from .alignment.filter_alignment import FilterAlignment
from .data.jwst_data import DataMetaInformation
from .data.loader.target_loader import TargetData
from .jwst_likelihoods import TargetLikelihoodProducts
from .jwst_response import TargetResponseInput, build_target_response
from .likelihood.likelihood import ShotNoise
from .likelihood.target_likelihood import build_target_likelihood
from .parse.jwst_response import SkyMetaInformation
from .parse.rotation_and_shift.coordinates_correction import CorrectionModel
from .parse.rotation_and_shift.rotation_and_shift import (
    rotation_and_shift_algorithm_config_factory,
)
from .parse.zero_flux_model import ZeroFluxPriorConfigs, match_band_key
from .plotting.residuals import ResidualPlottingInformation
from .psf.psf_operator import PsfGaussianBroadening, PsfKernelMix, mixed_kernel
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
    psf_mix : tuple[object, ProbabilityConfig | None] | None
        ``(second_psf, prior)``: the PSF becomes ``(1 - t) psf + t second_psf``
        (`PsfKernelMix`), so t = 0 is ``psf``. ``second_psf`` takes the forms
        of a kernel ``psf``, with its shape and unit sum; e.g. a broad halo as
        ``psf`` and a sharp core make t the core fraction. ``prior`` is
        uniform or delta, under ``{band}_psf_mix``; the kernel's negative flux
        at its bounds is logged. With ``prior`` None, the telescope block's
        ``psf_mix`` gives it. ``psf_broadening`` then broadens the mix. The
        t of a sample ``x`` is ``build_parametric_prior_from_prior_config(
        f"{band}_psf_mix", prior)(x)`` for a uniform prior; a delta's t is its
        value.
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
    psf_mix: tuple[object, ProbabilityConfig | None] | None = None

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
    ``target_data.std`` must be finite and positive on the unmasked pixels in
    either case; the WFPC2 loader gives NaN, so WFPC2 exposures raise (unless
    a caller replaces ``std``, which this check cannot see).

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
        ``correction_priors``, keyed by band or ``default``; with ``model:
        rshift`` each exposure also rotates by its ``rotation`` prior about
        the sky-grid centre), optional
        ``zero_flux`` and ``calibration`` (``default`` and band keys, one
        factor per exposure), and ``psf_broadening`` (same keys: the prior of
        the σ in arcsec of a Gaussian that broadens each exposure's kernel,
        see `PsfGaussianBroadening`, after any ``psf_mix``; without the key or at
        a delta of zero the kernels stay as given). Fixed parts, a delta prior or
        no key, are applied to the kernels once and make no parameter: an
        exposure with neither a learned weight nor a learned width keeps
        `PsfStatic`'s cached convolution, one with only a learned weight
        `PsfKernelMix`'s. ``psf_mix`` (same keys, matched against the
        exposures with a second kernel only) is the prior of the weight t of
        each such exposure whose ``Exposure.psf_mix`` gives none; an exposure
        that gives one as well raises, as does one with no prior. Band keys
        match in any case. Besides them, every
        one of these blocks takes prefix patterns: ``keck_nirc2_h_*`` covers
        every band starting ``keck_nirc2_h_``. A band's own key wins, then the
        longest pattern, then ``default``. A band key that matches no exposure
        raises; a pattern that matches none is logged.
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
    for e in exposures:
        # every loader gives a finite σ > 0 on unmasked pixels, except WFPC2's NaN
        std = np.asarray(e.target_data.std)[np.asarray(e.target_data.mask)]
        if not np.all(np.isfinite(std) & (std > 0)):
            raise ValueError(
                f"{e.band}: target_data.std is not a finite σ > 0 on the unmasked "
                "pixels. WFPC2 is not supported yet: it needs a noise model (σ² = "
                "a + b·max(R + sky, 0) from ATODGAIN and the read noise) and "
                "charge diffusion for subsampled Tiny Tim kernels"
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
    # band keys are matched case-insensitively, as in zero_flux and calibration
    correction_priors = {
        k.lower(): v for k, v in rotation_and_shift["correction_priors"].items()
    }
    zero_flux = ZeroFluxPriorConfigs.from_yaml_dict(telescope_cfg.get("zero_flux"))
    # same layout as zero_flux: a default and per-band priors
    calibration = ZeroFluxPriorConfigs.from_yaml_dict(telescope_cfg.get("calibration"))
    psf_broadening = ZeroFluxPriorConfigs.from_yaml_dict(
        telescope_cfg.get("psf_broadening")
    )
    psf_mix = ZeroFluxPriorConfigs.from_yaml_dict(telescope_cfg.get("psf_mix"))
    # a mistyped band key would silently fall back to the default prior
    reserved = {"default", "model", "shift_unit", "rotation_unit"}
    mixed = [e.band for e in exposures if e.psf_mix is not None]
    for name, keys, covered in (
        ("correction_priors", correction_priors, bands),
        ("zero_flux", zero_flux.names if zero_flux else {}, bands),
        ("calibration", calibration.names if calibration else {}, bands),
        ("psf_broadening", psf_broadening.names if psf_broadening else {}, bands),
        # only an exposure with a second kernel takes a weight
        ("psf_mix", psf_mix.names if psf_mix else {}, mixed),
    ):
        lower = [b.lower() for b in covered]
        # band keys hold no `*`, so `keck_*_h_*` could match no data set
        patterns = {k for k in keys if k.endswith("_*") and "*" not in k[:-1]}
        if unknown := set(keys) - patterns - set(lower) - reserved:
            raise ValueError(
                f"{name}: {sorted(unknown)} match no band of {covered}"
                + (", the exposures with a second kernel" if name == "psf_mix" else "")
            )
        # one fit fragment may serve data sets that lack a filter
        for k in sorted(patterns):
            if not any(match_band_key({k}, b) for b in lower):
                jft.logger.info(f"{name}: {k} matches no exposure of {covered}")
    if psf_mix and not mixed:
        jft.logger.info("psf_mix: no exposure names a second kernel")

    likelihoods = []
    plotting = ResidualPlottingInformation()
    for e in exposures:
        if e.psf_mix is not None:
            second, prior = e.psf_mix
            if (prior is None) == (psf_mix is None):
                found = "neither" if prior is None else "both"
                raise ValueError(
                    f"{e.band}: the psf_mix weight prior goes in the exposure or "
                    f"in the telescope block's psf_mix, found in {found}"
                )
            if prior is None:
                prior = psf_mix.get_name_setting_or_default(e.band)
                e = replace(e, psf_mix=(second, prior))
        target_data, psf_model = _with_psf(e)
        broadening = psf_broadening and psf_broadening.get_name_setting_or_default(
            e.band
        )
        # a width fixed at zero broadens nothing, so it keeps even a PSF model
        if (
            isinstance(broadening, DeltaPriorConfig)
            and broadening.transformation is None
            and not np.any(broadening.mean)
        ):
            broadening = None
        if e.psf_mix is not None or broadening is not None:
            if psf_model is not None:
                raise ValueError(
                    f"{e.band}: psf_mix or psf_broadening needs a kernel, the psf "
                    "is a model"
                )
            target_data, psf_model = _psf(e, target_data, broadening)
        correction_prior = FilterAlignment(e.band.lower()).load_correction_prior(
            correction_priors, number_of_observations=1
        )
        # the shift model ignores the rotation prior: only a fixed zero is no loss
        rotation = correction_prior.rotation
        if correction_prior.model == CorrectionModel.SHIFT and not (
            isinstance(rotation, DeltaPriorConfig) and not np.any(rotation.mean)
        ):
            raise ValueError(
                f"{e.band}: a rotation prior needs model rshift; under model "
                "shift give [delta, 0.0]"
            )
        correction = ShiftAndRotationCorrection(
            domain_key=e.band,
            correction_prior=correction_prior,
            # near the target a rotation barely moves it: rotation and shift decouple
            # the centre pixel, not `grid.spatial.center`, which may be NaN
            rotation_center=grid.spatial.indices_yx_to_world(
                *(np.array(grid.spatial.shape_yx) - 1) / 2
            ),
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


def _psf(
    e: Exposure, target_data: TargetData, broadening: ProbabilityConfig | None
) -> tuple[TargetData, jft.Model | None]:
    """The exposure's kernel mixed by ``e.psf_mix`` and broadened by a Gaussian
    whose σ in arcsec has the prior `broadening`, as `PsfStatic`'s kernel if
    nothing is learned, else as a PSF model."""
    sky = jft.ShapeWithDtype((1,) + target_data.subsample_centers[0].shape, float)
    kernels, weight, width = [target_data.psf], None, None
    if e.psf_mix is not None:
        second, prior = e.psf_mix
        kernels.append(_kernel(e, second))
        weight = _mix_weight(e, prior, *kernels)
    if broadening is not None:
        sigma = _parameter(f"{e.band}_psf_broadening", broadening)
        # PsfGaussianBroadening takes σ in pixels of the subsampled data grid
        pixel = e.meta.pixel_scale.to_value(u.arcsec) / target_data.subsample
        width = jft.Model(lambda x: sigma(x) / pixel, domain=sigma.domain)
    # fixed parts go into the kernels once
    if weight is not None and not weight.domain:
        kernels, weight = [mixed_kernel(*kernels, weight)({})], None
    if width is not None and not width.domain:
        kernels = [
            PsfGaussianBroadening(sky, k, width).broadened_kernel({}) for k in kernels
        ]
        width = None
    kernels = [np.asarray(k) for k in kernels]
    if width is None:
        if weight is None:
            return replace(target_data, psf=kernels[0]), None
        return target_data, PsfKernelMix(sky, *kernels, weight)
    kernel = kernels[0] if weight is None else mixed_kernel(*kernels, weight)
    return target_data, PsfGaussianBroadening(sky, kernel, width)


def _mix_weight(
    e: Exposure, prior: ProbabilityConfig, first: np.ndarray, second: np.ndarray
) -> jft.Model:
    """The weight t of ``e.psf_mix``; the prior must be uniform or delta."""
    # past [0, 1] the kernel turns negative where the kernels differ most: the
    # prior must bound t, and a bound is where the kernel is most negative
    if isinstance(prior, UniformPriorConfig) and prior.transformation is None:
        bounds = prior.min, prior.max
    elif isinstance(prior, DeltaPriorConfig) and prior.transformation is None:
        bounds = (prior.mean,)
    else:
        raise ValueError(
            f"{e.band}: the psf_mix weight needs a uniform or delta prior without "
            f"a transformation, got {prior}"
        )
    for t in bounds:
        kernel = np.asarray(
            mixed_kernel(first, second, jft.Model(lambda _: t, domain={}))({})
        )
        jft.logger.info(
            f"{e.band}: the psf_mix kernel at t = {t:g} has "
            f"{abs(kernel[kernel < 0].sum()):.2%} of its flux negative"
        )
    return _parameter(f"{e.band}_psf_mix", prior)


def _parameter(key: str, prior: ProbabilityConfig) -> jft.Model:
    """The prior of `key` as a model; a delta prior as a constant, no parameter."""
    model = build_parametric_prior_from_prior_config(key, prior, as_model=True)
    if isinstance(prior, DeltaPriorConfig):
        value = model({key: 0.0})
        return jft.Model(lambda _: value, domain={})
    return model


def _with_psf(e: Exposure) -> tuple[TargetData, jft.Model | None]:
    """The target data with the exposure's kernel, or with its PSF model."""
    if isinstance(e.psf, jft.Model):
        return e.target_data, e.psf
    return replace(e.target_data, psf=_kernel(e, e.psf)), None


def _kernel(e: Exposure, kernel) -> np.ndarray:
    """`kernel` as a (1, K, K) array, its sampling checked against the data."""
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
    return kernel.reshape((1,) + kernel.shape[-2:])
