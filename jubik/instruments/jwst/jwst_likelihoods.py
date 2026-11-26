from dataclasses import dataclass, replace
from functools import reduce
from typing import Union

import nifty.re as jft
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from nifty.re import logger

from ...grid import Grid
from ...likelihood import connect_likelihood_to_model
from ...models.distributions import build_parametric_prior_from_prior_config
from ...parse.models.distributions.distributions import (
    DeltaPriorConfig,
    UniformPriorConfig,
)
from ...sky_filter import SkyFilter
from .alignment.filter_alignment import FilterAlignment
from .config_handler import data_margin, get_grid_extension_from_config
from .data.loader.data_loader import (
    DataLoader,
    DataLoaderStarAlignment,
    DataLoaderTarget,
    load_data,
)
from .data.preloader.preloader import (
    Preloader,
    PreloaderSideEffects,
    preload_data,
)
from .data.jwst_information import JWST_FILTERS
from .throughput import jwst_filter_weights
from .jwst_response import TargetResponseInput, build_target_response
from .likelihood.alignment_likelihood import (
    AlignmentLikelihoodSideEffects,
    MultiFilterAlignmentLikelihoods,
    StarAlignmentResponseInput,
    build_star_alignment_likelihood,
)
from .likelihood.target_likelihood import (
    SingleTargetLikelihood,
    build_target_likelihood,
)
from .minimization.mask_hot_pixel_data import HotPixelMaskingData
from .parse.jwst_response import SkyMetaInformation
from .parse.parsing_step import ConfigParserJwst
from .parse.rotation_and_shift.coordinates_correction import CorrectionModel
from .parse.rotation_and_shift.rotation_and_shift import NufftConfig
from .plotting.alignment import MultiFilterAlignmentPlottingInformation
from .plotting.residuals import ResidualPlottingInformation
from .psf.psf_operator import PsfGaussianBroadening, PsfStatic
from .rotation_and_shift.coordinates_correction import ShiftAndRotationCorrection
from .variable_covariance.inverse_standard_deviation import (
    build_inverse_standard_deviation,
)


@dataclass
class TargetLikelihoodProducts:
    likelihoods: list[SingleTargetLikelihood]
    plotting: ResidualPlottingInformation
    sky_filter: SkyFilter
    hot_pixel_masking_data: HotPixelMaskingData | None = None

    @property
    def likelihood(self) -> jft.Likelihood | jft.Gaussian:
        likelihoods = (t.likelihood for t in self.likelihoods)
        likelihood = reduce(lambda x, y: x + y, likelihoods)
        return connect_likelihood_to_model(likelihood, self.sky_filter)


@dataclass
class MultiFilterAlignmentProducts:
    """Bundles the alignment likelihoods and plotting information over all
    filters; None-able via `from_optional`."""

    likelihood: MultiFilterAlignmentLikelihoods
    plotting: MultiFilterAlignmentPlottingInformation

    @classmethod
    def from_optional(
        cls,
        likelihood: MultiFilterAlignmentLikelihoods | None,
        plotting: MultiFilterAlignmentPlottingInformation,
    ) -> Union["MultiFilterAlignmentProducts", None]:
        if likelihood is None:
            return None
        return cls(likelihood=likelihood, plotting=plotting)


@dataclass
class JwstLikelihoodProducts:
    target: TargetLikelihoodProducts
    alignment: MultiFilterAlignmentProducts | None


def _prior_extent(prior) -> float:
    """The largest |value| a prior gives: its bounds, or |mean| + 3 sigma."""
    if isinstance(prior, UniformPriorConfig):
        return float(np.max(np.abs([prior.min, prior.max])))
    if isinstance(prior, DeltaPriorConfig):
        return float(np.max(np.abs(prior.mean)))
    return float(np.max(np.abs(prior.mean) + 3 * np.abs(prior.sigma)))


def _correction_reach(prior, grid: Grid, rotation_center: SkyCoord) -> float:
    """How far (arcsec) the shift, and for rshift the rotation about
    `rotation_center`, can move a subsample point."""
    reach = _prior_extent(prior.shift_in(u.arcsec))
    if prior.model == CorrectionModel.RSHIFT:
        corners = grid.spatial.world_corners()
        # one rotation centre per exposure: the farthest centre-corner pair
        far = np.max(
            [c.separation(rotation_center).to_value(u.arcsec) for c in corners]
        )
        reach += _prior_extent(prior.rotation_in(u.rad)) * far
    return reach


def _check_cutout_holds_sky(target_data, grid: Grid, filter: str, reach: float):
    """Raise if a subsample point on a cutout's outer ring lies within one sky pixel
    (the reach of the linear and the NUFFT interpolation) plus `reach` arcsec of the
    sky: the sky could then extend beyond the cutout and its light there be lost."""
    n_y, n_x = grid.spatial.shape_yx
    pad_y, pad_x = 1 + reach / grid.spatial.pixel_scales_yx.to_value(u.arcsec)
    for centers in target_data.subsample_centers:
        for edge in (centers[0], centers[-1], centers[:, 0], centers[:, -1]):
            y, x = grid.spatial.world_to_indices_yx(edge)
            near_y = (y >= -pad_y) & (y <= n_y - 1 + pad_y)
            if np.any(near_y & (x >= -pad_x) & (x <= n_x - 1 + pad_x)):
                raise ValueError(
                    f"{filter}: the sky, moved by up to {reach:.3g} arcsec by the "
                    "shift and rotation priors, reaches the data cutout's edge; "
                    "raise target.data_arcsec_extension (automatic unless set), or "
                    "set it to 'off' for the PSF-sized cutout"
                )


def _auto_margin(reach: float, grid: Grid, pixel_scale: u.Quantity) -> int:
    """Detector pixels to grow a cutout of the sky's corners by on each side, so it
    holds every data pixel with a subsample point on the sky moved by `reach` arcsec:
    the reach, the interpolation's sky pixel and the sky pixel `world_corners` leaves
    out on the low side, all along the sky's axes, so times sqrt(2) for a detector at
    any angle to them; then half a data pixel, rounded up."""
    # shortcut: sqrt(2) bounds |cos| + |sin| of the detector's angle to the sky grid,
    # up to 41% too wide on an aligned detector; use the angle from the data's WCS
    # if the cutout size matters
    sky = grid.spatial.pixel_scales_yx.to_value(u.arcsec).max()
    reach = np.sqrt(2) * (reach + 2 * sky)
    return int(np.ceil(reach / pixel_scale.to_value(u.arcsec) + 0.5))


#: JWST's primary mirror, the D in the optics' cutoff D / lambda (m)
JWST_DIAMETER = 6.5


def _band_limited(algorithm, filter: str, subsample: int, pixel_scale, grid: Grid):
    """`algorithm` with its NUFFT band limit resolved for `filter`, cycles per arcsec.

    k_c = D / lambda at the filter's blue half-power edge is the optics' cutoff, f_s =
    subsample / pixel the subsampled grid's frequency along a detector axis. Sky
    frequencies within k_c of a multiple of f_s alias into the passband; a band in
    [k_c, f_s - k_c] stops that and keeps the passband. "auto": none when not even
    the sky grid's corner frequency can alias, else f_s / 2 (the window's middle),
    kept below the sky grid's Nyquist frequency.
    """
    if not isinstance(algorithm, NufftConfig) or algorithm.band_limit is None:
        return algorithm
    blue = JWST_FILTERS[filter.upper()][3] * 1e-6  # m, the blue half-power edge
    k_c = JWST_DIAMETER / blue * np.pi / 648000  # cycles per arcsec
    f_s = subsample / pixel_scale.to_value(u.arcsec)
    nyquist = 0.5 / grid.spatial.pixel_scales_yx.to_value(u.arcsec)
    band = algorithm.band_limit
    if band == "auto":
        if np.hypot(*nyquist) + k_c <= f_s:
            return replace(algorithm, band_limit=None)
        band = min(f_s / 2, 0.999 * nyquist.min())
    if not k_c <= band <= f_s - k_c:
        raise ValueError(
            f"{filter}: a NUFFT band_limit of {band:.4g} cycles/arcsec must lie "
            f"between the optics' cutoff D/lambda {k_c:.4g} and {f_s - k_c:.4g} "
            f"(subsample {subsample}: {f_s:.4g} less the cutoff), to keep the passband "
            "and stop aliasing; raise target.subsample, or set "
            "rotation_and_shift.nufft.band_limit: off"
        )
    return replace(algorithm, band_limit=float(band))


def _data_inside_sky(target_data, grid: Grid, inset: float) -> np.ndarray:
    """True for the data pixels whose subsample points all lie at least `inset`
    arcsec inside the sky box, whose edges are half a sky pixel beyond the outer
    pixel centres."""
    n_y, n_x = grid.spatial.shape_yx
    low_y, low_x = inset / grid.spatial.pixel_scales_yx.to_value(u.arcsec) - 0.5
    s = target_data.subsample
    inside = []
    for centers in target_data.subsample_centers:
        y, x = grid.spatial.world_to_indices_yx(centers)
        points = (y >= low_y) & (y <= n_y - 1 - low_y)
        points &= (x >= low_x) & (x <= n_x - 1 - low_x)
        h, w = points.shape
        inside.append(points.reshape(h // s, s, w // s, s).all(axis=(1, 3)))
    return np.stack(inside)


def build_jwst_likelihoods(
    cfg: dict,
    grid: Grid,
    sky_domain: dict[str, jft.ShapeWithDtype],
    sky_unit: u.Unit = u.Unit("MJy") / u.Unit("sr"),
    files_key: str = "files",
    telescope_key: str = "telescope",
) -> JwstLikelihoodProducts:
    """Build the jwst likelihood_target according to the config and grid."""

    assert isinstance(sky_domain, dict) and len(sky_domain) == 1, (
        "Only works with a single sky key since the likelihood has learnable parameters"
    )

    ((sky_key, sky_leaf),) = sky_domain.items()
    sky_filter = SkyFilter(
        grid,
        {
            name: jwst_filter_weights(
                grid.spectral, name, cfg[files_key].get("throughputs")
            )
            for name in cfg[files_key]["filter"]
        },
        sky_key=sky_key,
        dtype=sky_leaf.dtype,
    )
    if sky_filter.domain[sky_key].shape != sky_leaf.shape:
        raise ValueError(
            f"sky_domain shape {sky_leaf.shape} does not match the grid "
            f"shape {sky_filter.domain[sky_key].shape}"
        )

    # Parsing
    sky_meta = SkyMetaInformation(
        grid_extension=get_grid_extension_from_config(cfg[telescope_key], grid),
        unit=sky_unit,
        dvol=grid.spatial.dvol,
    )
    cfg_parser = ConfigParserJwst.from_yaml_dict(
        cfg, telescope_key=telescope_key, files_key=files_key
    )
    margin = data_margin(cfg[telescope_key])
    # only mode constant reads zero sky outside the grid, which the margin check assumes
    if (
        margin is not None
        and cfg_parser.rotation_and_shift_algorithm.mode != "constant"
    ):
        raise ValueError(
            "target.data_arcsec_extension (automatic unless 'off') needs "
            "rotation_and_shift mode 'constant'; set it to 'off' for the PSF-sized "
            "cutout"
        )
    # fit only data this far (arcsec) inside the sky box, so that light from beyond
    # the box near the fitted data's edge can come from sky that is not fitted directly
    data_inset = cfg[telescope_key].get("target", {}).get("data_arcsec_inset")
    if data_inset is not None and data_inset < 0:
        raise ValueError(f"target.data_arcsec_inset must be >= 0, got {data_inset}")

    target_plotting = ResidualPlottingInformation()
    target_filter_likelihoods = []
    alignment_plotting = (
        MultiFilterAlignmentPlottingInformation()
        if cfg_parser.star_alignment_config is not None
        else None
    )
    alignment_likelihoods = (
        MultiFilterAlignmentLikelihoods()
        if cfg_parser.star_alignment_config is not None
        else None
    )

    hot_pixel_masking_data = HotPixelMaskingData()

    for filter, filepaths in cfg_parser.data_loader.paths.items():
        filter_alignment = FilterAlignment(filter_name=filter)

        preload_results = preload_data(
            filepaths=filepaths,
            filter_name=filter,
            preloader=Preloader(
                grid_corners=grid.spatial.world_corners(
                    extension_value=sky_meta.grid_extension
                ),
                star_alignment_config=cfg_parser.star_alignment_config,
            ),
            side_effects=PreloaderSideEffects(filter_alignment=filter_alignment),
        )

        correction_prior = filter_alignment.load_correction_prior(
            cfg[telescope_key]["rotation_and_shift"]["correction_priors"],
            number_of_observations=len(filepaths),
        )
        rotation_center = SkyCoord(filter_alignment.boresight)
        reach = _correction_reach(correction_prior, grid, rotation_center)
        if margin == "auto":
            m = _auto_margin(reach, grid, preload_results.filter_meta.pixel_scale)
            preload_results.target_bounds.cast_to_numpy()
            preload_results.target_bounds.adjust_bounds(np.array([-m, m, -m, m]))

        if filter in cfg_parser.data_bounds_adjust:
            preload_results.target_bounds.cast_to_numpy()
            preload_results.target_bounds.adjust_bounds(
                cfg_parser.data_bounds_adjust.get(filter)
            )
            preload_results.target_bounds.align_bounds()

        dataload_results = load_data(
            filepaths=filepaths,
            data_loader=DataLoader(
                target=DataLoaderTarget.from_optional(
                    grid=grid,
                    data_bounds=preload_results.target_bounds,
                    subsample=cfg_parser.subsample_target,
                    nan_mask_loader=cfg_parser.nan_masks.get(
                        filter, preload_results.target_bounds.bounds
                    ),
                ),
                psf_kernel_configs=cfg_parser.psf_kernel_configs,
                star_alignment=DataLoaderStarAlignment.from_optional(
                    config=cfg_parser.star_alignment_config,
                    tables=preload_results.star_tables,
                ),
                corner_masks=cfg_parser.corner_masks(filter),
            ),
        )

        hot_pixel_masking_data.append_information(
            filter=filter,
            nan_mask=dataload_results.target_data.nan_mask,
            bounds=preload_results.target_bounds.bounds,
        )

        # Constructing the Likelihood
        shift_and_rotation_correction = ShiftAndRotationCorrection(
            domain_key=filter,
            correction_prior=correction_prior,
            rotation_center=rotation_center,
        )

        if margin is not None:
            _check_cutout_holds_sky(dataload_results.target_data, grid, filter, reach)
        if data_inset is not None:
            target_data = dataload_results.target_data
            kept = int(target_data.mask.sum())
            # the inset holds after the correction priors move the data by `reach`
            target_data.mask = target_data.mask & _data_inside_sky(
                target_data, grid, data_inset + reach
            )
            if not target_data.mask.any():
                raise ValueError(
                    f"{filter}: target.data_arcsec_inset {data_inset} plus the "
                    f"priors' reach {reach:.3g} arcsec masks every data pixel"
                )
            logger.info(
                f"{filter}: target.data_arcsec_inset {data_inset} + prior reach "
                f"{reach:.3g} arcsec masks {kept - int(target_data.mask.sum())} of "
                f"{kept} unmasked data pixels"
            )
        logger.debug(
            f"{filter}: {int(dataload_results.target_data.mask.sum())} unmasked data pixels"
        )
        algorithm = _band_limited(
            cfg_parser.rotation_and_shift_algorithm,
            filter,
            cfg_parser.subsample_target,
            preload_results.filter_meta.pixel_scale,
            grid,
        )

        likelihood_target: SingleTargetLikelihood = build_target_likelihood(
            filter_name=filter,
            response=build_target_response(
                input_config=TargetResponseInput(
                    filter_name=filter,
                    grid=grid,
                    sky_filter=sky_filter,
                    target_data=dataload_results.target_data,
                    psf_model=_broadened_psf(
                        filter,
                        cfg_parser.psf_broadening,
                        dataload_results.target_data,
                        preload_results.filter_meta.pixel_scale,
                    ),
                    filter_meta=preload_results.filter_meta,
                    sky_meta=sky_meta,
                    rotation_and_shift_algorithm=algorithm,
                    zero_flux_prior_configs=cfg_parser.zero_flux_prior_configs,
                    shift_and_rotation_correction=shift_and_rotation_correction,
                )
            ),
            target_data=dataload_results.target_data,
            inverse_std_builder=build_inverse_standard_deviation(
                filter_name=filter,
                config=cfg_parser.variable_covariance_config,
            ),
        )

        target_plotting.append_information(
            filter=filter,
            data=dataload_results.target_data.data,
            std=dataload_results.target_data.std,
            mask=dataload_results.target_data.mask,
            builder=likelihood_target.builder,
            meta=preload_results.filter_meta,
        )

        target_filter_likelihoods.append(likelihood_target)

        if cfg_parser.star_alignment_config is not None:
            filter_alignment_likelihood = build_star_alignment_likelihood(
                response=StarAlignmentResponseInput(
                    filter_name=filter,
                    filter_meta=preload_results.filter_meta,
                    sky_meta=sky_meta,
                    star_tables=preload_results.star_tables,
                    stars_data=dataload_results.stars_data,
                    star_light_prior=cfg_parser.star_alignment_config.star_light_prior,
                    shift_and_rotation_correction=shift_and_rotation_correction,
                    zero_flux_prior_configs=cfg_parser.zero_flux_prior_configs,
                    fit_star_positions=cfg_parser.star_alignment_config.fit_star_positions,
                    grid_wcs=grid.spatial,
                ),
                side_effect=AlignmentLikelihoodSideEffects(plotting=alignment_plotting),
            )
            alignment_likelihoods.likelihoods.append(filter_alignment_likelihood)

    return JwstLikelihoodProducts(
        target=TargetLikelihoodProducts(
            likelihoods=target_filter_likelihoods,
            plotting=target_plotting,
            sky_filter=sky_filter,
            hot_pixel_masking_data=hot_pixel_masking_data,
        ),
        alignment=MultiFilterAlignmentProducts.from_optional(
            likelihood=alignment_likelihoods,
            plotting=alignment_plotting,
        ),
    )


def _broadened_psf(
    filter: str, configs, target_data, pixel_scale: u.Quantity
) -> jft.Model | None:
    """The PSF of `filter`'s target response with each exposure's kernel broadened by
    a Gaussian (`PsfGaussianBroadening`) whose σ in arcsec has the filter's
    ``psf_broadening`` prior in `configs`: one σ per exposure, the parameter
    ``{filter}_psf_broadening`` of shape (n_exposures,). The gradient vanishes at
    σ = 0, so a learned σ needs a positive prior such as a lognormal.

    None, the kernels as loaded, without a prior or at a delta of zero. Another
    delta broadens every kernel once, for `PsfStatic`'s cached convolution. The
    kernels are taken as loaded, with any ``psf.correction`` and ``psf.pixel_box``.
    """
    prior = configs and configs.get_name_setting_or_default(filter)
    # a width fixed at zero broadens nothing
    if prior is None or (
        isinstance(prior, DeltaPriorConfig)
        and prior.transformation is None
        and not np.any(prior.mean)
    ):
        return None
    kernels = np.asarray(target_data.psf)
    n = kernels.shape[0]
    sky = jft.ShapeWithDtype((n,) + target_data.subsample_centers[0].shape, float)
    key = f"{filter}_psf_broadening"
    sigma = build_parametric_prior_from_prior_config(key, prior, (n,), as_model=True)
    # PsfGaussianBroadening takes σ in pixels of the subsampled data grid
    pixel = pixel_scale.to_value(u.arcsec) / target_data.subsample
    width = jft.Model(lambda x: sigma(x) / pixel, domain=sigma.domain)
    psf = PsfGaussianBroadening(sky, kernels, width)
    if isinstance(prior, DeltaPriorConfig):
        return PsfStatic(sky, np.asarray(psf.broadened_kernel({key: np.zeros(n)})))
    return psf
