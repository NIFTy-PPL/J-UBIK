from dataclasses import dataclass, field, replace
from collections import Counter
from functools import reduce
from itertools import chain
from pathlib import Path
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
from .alignment.star_model import gaia_positions
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
from .data.jwst_data import JwstData
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
from .psf.psf_correction import (
    STAR_PAD,
    StarStamp,
    joint_correction,
    load_star_stamps,
)
from .psf.psf_operator import PsfGaussianBroadening, PsfStatic
from .rotation_and_shift.coordinates_correction import ShiftAndRotationCorrection
from .variable_covariance.inverse_standard_deviation import (
    build_inverse_standard_deviation,
)


@dataclass
class TargetLikelihoodProducts:
    """`likelihood` sums the filters' target likelihoods and the stars'
    likelihoods of ``psf.joint_stars`` (`joint_stars`), which share the
    filters' learned PSF corrections."""

    likelihoods: list[SingleTargetLikelihood]
    plotting: ResidualPlottingInformation
    sky_filter: SkyFilter
    hot_pixel_masking_data: HotPixelMaskingData | None = None
    joint_stars: list[jft.Likelihood] = field(default_factory=list)

    @property
    def likelihood(self) -> jft.Likelihood | jft.Gaussian:
        likelihoods = (t.likelihood for t in self.likelihoods)
        likelihoods = chain(likelihoods, self.joint_stars)
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
    """Build the jwst likelihood_target according to the config and grid.

    An optional ``calibration`` block in the telescope config (``default`` and
    filter keys, as ``zero_flux``) gives each exposure of a filter its own
    multiplicative flux factor, ``{filter}_target_calibration`` of shape
    ``(n_exposures, 1, 1)``; without it the factor is 1. The factors are taken
    relative to their geometric mean (the filter's sky carries the common factor)
    and need a lognormal prior; its mean drops out as a common factor, and only
    std / mean, which sets the log-space width, matters.
    """

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
    star_likelihoods = []
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
        psf_config = cfg_parser.psf_kernel_configs
        joint = psf_config.joint_stars.get(filter.lower())
        if joint is not None:  # the learned correction adds the box after itself
            psf_config = replace(psf_config, pixel_box=False)

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
                psf_kernel_configs=psf_config,
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

        kappa = _kappa(filter, cfg_parser.psf_kernel_configs)
        psf_model = _broadened_psf(
            filter,
            cfg_parser.psf_broadening,
            dataload_results.target_data,
            preload_results.filter_meta.pixel_scale,
        )
        if joint is not None:
            gaia = None
            if cfg_parser.star_alignment_config is not None:
                gaia = _GaiaStars(
                    tables=preload_results.star_tables,
                    data=dataload_results.stars_data,
                    filepaths=filepaths,
                    correction=shift_and_rotation_correction,
                    grid_wcs=grid.spatial,
                    fit_position=cfg_parser.star_alignment_config.fit_star_positions,
                    tolerance=reach
                    + preload_results.filter_meta.pixel_scale.to_value(u.arcsec),
                )
            psf_model, stars = _joint_stars(
                filter,
                joint,
                cfg_parser.psf_kernel_configs.pixel_box,
                kappa,
                dataload_results.target_data,
                gaia,
            )
            star_likelihoods += stars

        likelihood_target: SingleTargetLikelihood = build_target_likelihood(
            filter_name=filter,
            response=build_target_response(
                input_config=TargetResponseInput(
                    filter_name=filter,
                    grid=grid,
                    sky_filter=sky_filter,
                    target_data=dataload_results.target_data,
                    psf_model=psf_model,
                    filter_meta=preload_results.filter_meta,
                    sky_meta=sky_meta,
                    rotation_and_shift_algorithm=algorithm,
                    zero_flux_prior_configs=cfg_parser.zero_flux_prior_configs,
                    shift_and_rotation_correction=shift_and_rotation_correction,
                    calibration_prior=(
                        None
                        if cfg_parser.calibration_prior_configs is None
                        else cfg_parser.calibration_prior_configs.get_name_setting_or_default(
                            filter
                        )
                    ),
                    calibration_relative=True,
                    brighter_fatter=_brighter_fatter(kappa, filepaths),
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

        # a joint_stars filter models its Gaia stars in `stars`
        if cfg_parser.star_alignment_config is not None and joint is None:
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
            joint_stars=star_likelihoods,
        ),
        alignment=MultiFilterAlignmentProducts.from_optional(
            likelihood=(
                alignment_likelihoods
                if alignment_likelihoods and alignment_likelihoods.likelihoods
                else None
            ),
            plotting=alignment_plotting,
        ),
    )


def _kappa(filter: str, psf_config) -> float | jft.Model | None:
    """`filter`'s kappa per DN per integration: ``psf.brighter_fatter``'s, a model
    of ``psf.joint_stars``' log-normal prior (``{filter}_psf_brighter_fatter``),
    or None."""
    joint = psf_config.joint_stars.get(filter.lower())
    if joint is not None and joint.brighter_fatter is not None:
        name = f"{filter}_psf_brighter_fatter"
        return jft.LogNormalPrior(*joint.brighter_fatter, name=name, shape=())
    return psf_config.brighter_fatter.get(filter.lower())


def _brighter_fatter(kappa, filepaths) -> np.ndarray | jft.Model | None:
    """`kappa` (see `_kappa`) per MJy/sr of each exposure, (n_exposures, 1, 1):
    kappa per DN per integration times EFFINTTM / PHOTMJSR, the DN per
    integration of one MJy/sr. None for None."""
    if kappa is None:
        return None
    dn_per_unit = _dn_per_unit(filepaths)[:, None, None]
    if isinstance(kappa, jft.Model):
        return jft.Model(lambda x: kappa(x) * dn_per_unit, domain=kappa.domain)
    return kappa * dn_per_unit


def _dn_per_unit(filepaths) -> np.ndarray:
    """EFFINTTM / PHOTMJSR of each exposure, in `TargetData`'s order: the DN per
    integration of one MJy/sr."""
    dn_per_unit = []
    for path in sorted(filepaths, key=lambda p: p.index):  # TargetData's order
        data = JwstData(path.path)
        time = data.dm.meta.exposure.integration_time
        photmjsr = data.dm.meta.photometry.conversion_megajanskys
        if data.meta.unit != u.MJy / u.sr or not time or not photmjsr:
            raise ValueError(
                f"psf brighter_fatter needs data in MJy/sr with EFFINTTM and "
                f"PHOTMJSR; {path.path}: {data.meta.unit}, EFFINTTM {time}, "
                f"PHOTMJSR {photmjsr}"
            )
        dn_per_unit.append(time / photmjsr)
    return np.array(dn_per_unit)


@dataclass
class _GaiaStars:
    """What `_joint_stars` needs of a filter's ``gaia_alignment``: the Gaia
    tables and cutouts, the filter's files and shift, and how far (arcsec) a
    stamp's centre may lie from a Gaia star's catalog position to be it."""

    tables: object  # StarTables
    data: object  # StarsData
    filepaths: list
    correction: ShiftAndRotationCorrection
    grid_wcs: object
    fit_position: bool
    tolerance: float


def _joint_stars(
    filter: str, joint, pixel_box: bool, kappa, target_data, gaia=None
) -> tuple[jft.Model, list[jft.Likelihood]]:
    """The PSF of `filter`'s target response and its stars' likelihoods, sharing
    the correction learned on all (`joint_correction`, latents named
    ``{filter}_...``). With `gaia` (`_GaiaStars`) every Gaia star in the files
    is a star of the fit once: a stamp at a Gaia star's position sits there,
    moved by its exposure's shift (and astrometry), and a Gaia star without a
    stamp gets its gaia_alignment cutout, ``{filter}_gaia_...``."""
    stamps, subsample, extra = load_star_stamps(joint.stamps)
    if subsample != target_data.subsample:
        raise ValueError(
            f"psf joint_stars {filter}: the stamps' kernels in {joint.stamps} are "
            f"at subsample {subsample}, the target's at {target_data.subsample}"
        )
    kernels = np.asarray(target_data.psf)
    sky = jft.ShapeWithDtype(
        (kernels.shape[0],) + target_data.subsample_centers[0].shape, float
    )
    rules = joint.exclude
    if rules is not None and rules.max_peak is not None:
        too_bright = [s.fluence > rules.max_peak for s in stamps]
        why = f"a peak above {rules.max_peak:g} DN/int"
        stamps, extra = _drop_stamps(filter, stamps, extra, too_bright, why)
    positions, more = None, ()
    if gaia is not None:
        dn = None
        if kappa is not None or joint.fluence is not None or rules is not None:
            dn = _dn_per_unit(gaia.filepaths)
        stamps, positions, more = _tie_to_gaia(
            filter, joint, stamps, extra, subsample, gaia, dn
        )
    return joint_correction(
        stamps,
        subsample,
        kernels,
        sky,
        radius=joint.radius,
        fluctuations=joint.fluctuations,
        loglogavgslope=joint.loglogavgslope,
        width=joint.width,
        fluence=joint.fluence,
        orthogonal=joint.orthogonal,
        pixel_box=pixel_box,
        brighter_fatter=kappa,
        prefix=f"{filter}_",
        positions=positions,
        more_stars=more,
    )


def _drop_stamps(filter, stamps, extra, drop, why: str):
    """`stamps` and the arrays of `extra` without those where `drop`, logged
    with `why`."""
    drop = np.asarray(drop, bool)
    if not drop.any():
        return stamps, extra
    names = [
        f"{e} ({x}, {y})" if {"exposure", "x", "y"} <= set(extra) else f"#{i}"
        for i, e, x, y in zip(
            range(len(stamps)),
            extra.get("exposure", [None] * len(stamps)),
            extra.get("x", [None] * len(stamps)),
            extra.get("y", [None] * len(stamps)),
        )
    ]
    dropped = [n for n, d in zip(names, drop) if d]
    jft.logger.warning(f"{filter}: stamps excluded, {why}: {dropped}")
    if drop.all():
        raise ValueError(f"psf joint_stars {filter}: every stamp is excluded, {why}")
    return [s for s, d in zip(stamps, drop) if not d], {
        k: v[~drop] for k, v in extra.items()
    }


#: `StarExclusion`'s rules on Gaia columns: (rule, column, name)
_GAIA_RULES = (
    ("ruwe", "ruwe", "RUWE"),
    ("excess_noise_sig", "astrometric_excess_noise_sig", "excess noise significance"),
    ("galaxy", "classprob_dsc_combmod_galaxy", "galaxy probability"),
)


def _gaia_exclusion(filter, rules, table) -> dict[int, list[str]]:
    """The Gaia stars of `table` that `rules` drop, with why."""
    reasons = {}
    for rule, column, name in _GAIA_RULES:
        limit = getattr(rules, rule)
        if limit is None:
            continue
        if column not in table.colnames:
            jft.logger.info(f"{filter}: no Gaia {column}, exclude.{rule} skipped")
            continue
        values = np.asarray(np.ma.filled(table[column], np.nan), float)
        for source, value in zip(table["SOURCE_ID"], values):
            if value > limit:
                reasons.setdefault(int(source), []).append(f"{name} {value:.3g}")
    return reasons


def _tie_to_gaia(filter, joint, stamps, extra, subsample, gaia: _GaiaStars, dn):
    """The stamps left and `joint_correction`'s ``positions`` and
    ``more_stars`` for `filter`'s Gaia stars (see `_joint_stars`).

    A Gaia star in an exposure is a stamp of that file (`extra`'s
    ``exposure``) whose centre (``ra``, ``dec``) lies within `gaia.tolerance`
    of its catalog position at the exposure's epoch, the nearest pairs first;
    the stamp's centre pixel (``x``, ``y``) places that position in it. With
    ``joint.exclude`` a Gaia star its rules drop goes with its stamps, and a
    Gaia cutout they drop goes alone. `dn` is each exposure's DN per
    integration per data unit (`_dn_per_unit`), or None."""
    if missing := {"exposure", "x", "y", "ra", "dec"} - set(extra):
        raise ValueError(
            f"psf joint_stars {filter}: gaia_alignment needs the stamps' "
            f"{sorted(missing)} in {joint.stamps} (`save_star_stamps`)"
        )
    index = {Path(f.path).name: f.index for f in gaia.filepaths}
    exposure = np.array([index.get(str(n), -1) for n in extra["exposure"]])
    if outside := sorted({str(n) for n in extra["exposure"][exposure < 0]}):
        jft.logger.warning(
            f"{filter}: stamps of {outside}, no file of the fit, keep free "
            "positions; if one is a fit file under another name, its Gaia stars "
            "are modelled twice"
        )
    centres = SkyCoord(extra["ra"], extra["dec"], unit="deg")
    times = gaia.tables.observation_times
    frames, pairs = [], []  # (star, frame); (separation, stamp, frames index)
    for star in gaia.tables.get_stars():
        if star.id not in gaia.data:
            continue
        for k, e in enumerate(gaia.data[star.id].observation_ids):
            here = np.flatnonzero(exposure == e)
            sep = centres[here].separation(star.position_at(times[e])).arcsec
            pairs += [
                (d, j, len(frames)) for d, j in zip(sep, here) if d <= gaia.tolerance
            ]
            frames.append((star, k))
    tied, taken = {}, set()  # stamp -> (star, frame); the frames with a stamp
    for _, j, f in sorted(pairs):  # each stamp and each frame once
        if j not in tied and f not in taken:
            tied[j] = frames[f]
            taken.add(f)
    table = gaia.tables.join_tables()
    rules = joint.exclude
    excluded = {} if rules is None else _gaia_exclusion(filter, rules, table)
    gone = [j in tied and int(tied[j][0].id) in excluded for j in range(len(stamps))]
    kept = np.cumsum(np.logical_not(gone)) - 1  # each stamp's index once dropped
    tied = {int(kept[j]): v for j, v in tied.items() if not gone[j]}
    why = "at a Gaia star exclude drops"
    stamps, extra = _drop_stamps(filter, stamps, extra, gone, why)
    # a cutout masked whole (e.g. by a corner mask) holds no star
    alone = [
        (star, k)
        for f, (star, k) in enumerate(frames)
        if f not in taken
        and gaia.data[star.id].mask[k].any()
        and int(star.id) not in excluded
    ]
    if not frames:
        jft.logger.warning(
            f"{filter}: gaia_alignment finds no star in the fit's files; only the "
            "lens pins the shifts"
        )
    if "parallax" in table.colnames:
        near = np.asarray(np.ma.filled(table["parallax"], 0.0), float) >= 1.0
        if ids := sorted(map(int, set(table["SOURCE_ID"][near]) & set(gaia.data))):
            jft.logger.warning(
                f"{filter}: Gaia stars {ids} have a parallax of 1 mas or more, "
                "which their positions leave out"
            )

    def entry(star, k, pixel):
        frames = gaia.data[star.id]
        jacobian = frames.sky_to_data[k].to_value(u.pix / u.arcsec)
        return star, int(frames.observation_ids[k]), pixel, jacobian

    def fit_positions(entries):
        return gaia_positions(
            entries, gaia.correction, gaia.grid_wcs, times, gaia.fit_position
        )

    positions = None
    if tied:
        h = stamps[0].data.shape[-1] // 2
        entries = []
        for j, (star, k) in sorted(tied.items()):
            # the catalog position from the stamp's corner, in subsampled pixels
            xy = gaia.data[star.id].star_in_detector_pixels[k]
            corner = np.array([extra["x"][j], extra["y"][j]]) - h
            entries.append(entry(star, k, (xy - corner + 0.5) * subsample - 0.5))
        positions = (np.array(sorted(tied)), fit_positions(entries))
    cutouts, entries, suspect, dropped = [], [], set(), {}
    # in data units, about the star fit's peak cut; NaN without dn_per_unit
    brightest = np.array([s.fluence / (s.dn_per_unit or np.nan) for s in stamps])
    brightest = np.nanmax(brightest) if np.isfinite(brightest).any() else np.nan
    for star, k in alone:
        frames = gaia.data[star.id]
        data, mask = frames.data[k], frames.mask[k]
        e = int(frames.observation_ids[k])
        unit = None if dn is None else float(dn[e])
        edge = np.ones(data.shape, bool)
        edge[1:-1, 1:-1] = False
        peak = np.max(data[mask]) - np.median(
            data[mask & edge] if (mask & edge).any() else data[mask]
        )
        fluence = (
            0.0 if unit is None else peak * unit
        )  # as psf_check's, per integration
        inside = (frames.star_in_subsampled_pixels[k] + 0.5) / subsample - 0.5
        x, y = np.round(inside).astype(int)
        why = []
        if rules is not None and rules.saturated:
            if frames.saturated[k] or not mask[y, x]:
                why.append("saturated")
        if rules is not None and rules.max_peak is not None:
            if fluence > rules.max_peak:
                why.append(f"a peak above {rules.max_peak:g} DN/int")
        dropped.setdefault(int(star.id), Counter()).update(why)
        if why:
            continue
        cutouts.append(
            StarStamp(data, frames.std[k], mask, frames.psf[k], fluence, unit)
        )
        entries.append(entry(star, k, frames.star_in_subsampled_pixels[k]))
        if not mask[y, x] or 0 < brightest < peak:
            suspect.add(int(star.id))
    for star, counts in dropped.items():  # per frame: "<why> in <n> of <all> frames"
        n = len(gaia.data[star].observation_ids)
        excluded.setdefault(star, []).extend(
            f"{r} in {c} of {n} frames" for r, c in counts.items()
        )
    if logged := {s: r for s, r in excluded.items() if s in gaia.data and r}:
        jft.logger.warning(
            f"{filter}: Gaia stars excluded: "
            + "; ".join(f"{s}: {', '.join(r)}" for s, r in sorted(logged.items()))
        )
    remain = {int(star.id) for star, _ in tied.values()}
    remain |= {int(star.id) for star, *_ in entries}
    jft.logger.info(
        f"{filter}: {len(remain)} Gaia stars remain {sorted(remain)}: {len(tied)} star "
        f"stamps at Gaia stars, {len(stamps) - len(tied)} free, {len(cutouts)} Gaia "
        f"cutouts without a stamp (Gaia stars "
        f"{sorted({int(star.id) for star, *_ in entries})})"
    )
    if frames and not remain:
        jft.logger.warning(
            f"{filter}: no Gaia star remains after exclude; only the lens pins the "
            "shifts"
        )
    if suspect:
        jft.logger.warning(
            f"{filter}: Gaia stars {sorted(suspect)} have no stamp and are brighter "
            "than every stamp or masked at their centre (saturated?), yet shape the "
            "PSF and kappa; gaia_alignment's exclude_source_id drops them"
        )
    if not cutouts:
        return stamps, positions, ()
    pixels = np.array([p for *_, p, _ in entries])
    size = cutouts[0].data.shape[-1]
    # the stars sit off the middle of edge cutouts: the canvas reaches past them
    far = np.abs((pixels + 0.5) / subsample - 0.5 - size // 2).max()
    pad = STAR_PAD + int(np.ceil(far))
    group = (
        cutouts,
        (np.arange(len(cutouts)), fit_positions(entries)),
        f"{filter}_gaia_",
        pad,
    )
    return stamps, positions, (group,)


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
