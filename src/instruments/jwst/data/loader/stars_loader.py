from collections import UserDict
from dataclasses import dataclass, field

import numpy as np
from astropy import units as u
from nifty8.re import logger

from ...alignment.star_alignment import StarTables
from ...parse.alignment.star_alignment import StarAlignmentConfig
from ...parse.data.data_loader import IndexAndPath
from ...parse.jwst_psf import JwstPsfKernelConfig
from ...parse.masking.data_mask import ExtraMasks
from ...psf.jwst_kernel import load_psf_kernel
from ..jwst_data import JwstData
from .cutout import DataCutout

# ---------------------
# Loading
# ---------------------


@dataclass(slots=True)
class SingleStarBundle:
    cutout: DataCutout
    star_in_subsampled_pixels: tuple[float, float]
    observation_id: int


class StarsBundle(UserDict[int, SingleStarBundle]):
    def __init__(self, index: int, mapping: dict[int, SingleStarBundle] | None = None):
        self.index = index
        self.data = mapping if mapping is not None else {}


def load_one_stars_bundle(
    index: int,
    jwst_data: JwstData,
    star_tables: StarTables,
    star_alignment_config: StarAlignmentConfig,
    extra_masks: ExtraMasks,
    psf_kernel_configs: JwstPsfKernelConfig,
) -> StarsBundle:
    fov_pixel = (
        star_alignment_config.fov.to(u.arcsec) / jwst_data.meta.pixel_scale.to(u.arcsec)
    ).value
    fov_pixel = np.array((int(np.round(fov_pixel)),) * 2)
    if (fov_pixel % 2).sum() == 0:
        fov_pixel += 1

    star_bundles = StarsBundle(index=index)

    # logger.info("THIS SHOULDN't APPEAR DELETE ME!")
    # if False:
    #     from ...alignment.utils import some_evaluation
    #
    #     some_evaluation(
    #         index, jwst_data, star_tables, image_kwargs=dict(vmin=0.05, vmax=580)
    #     )

    for ii, star in enumerate(star_tables.get_stars(index)):
        bounding_indices = star.bounding_indices(jwst_data, fov_pixel)
        data, mask, std = jwst_data.bounding_data_mask_std_by_bounding_indices(
            row_minmax_column_minmax=bounding_indices,
            additional_masks_corners=extra_masks,
        )
        # check that data is not completely empty
        if np.all(np.isnan(data)):
            continue

        psf = load_psf_kernel(
            jwst_data=jwst_data,
            subsample=star_alignment_config.subsample,
            target_center=star.position,
            config_parameters=psf_kernel_configs,
        )

        star_in_subsampled_pixels = star.pixel_position_in_subsampled_data(
            jwst_data.wcs,
            min_row=bounding_indices[0],
            min_column=bounding_indices[2],
            subsample_factor=star_alignment_config.subsample,
        )

        star_bundles[star.id] = SingleStarBundle(
            cutout=DataCutout(data=data, mask=mask, std=std, psf=psf),
            star_in_subsampled_pixels=star_in_subsampled_pixels,
            observation_id=index,
        )

    return star_bundles


def load_one_stars_bundle_from_filepath(
    filepath: IndexAndPath,
    star_tables: StarTables,
    star_alignment_config: StarAlignmentConfig,
    extra_masks: ExtraMasks,
    psf_kernel_configs: JwstPsfKernelConfig,
) -> StarsBundle:
    jwst_data = JwstData(filepath.path)
    return load_one_stars_bundle(
        index=filepath.index,
        jwst_data=jwst_data,
        star_tables=star_tables,
        star_alignment_config=star_alignment_config,
        extra_masks=extra_masks,
        psf_kernel_configs=psf_kernel_configs,
    )


# ---------------------
# Products
# ---------------------


# ------------------------------------------------------------------
# 1) Per-star container (one entry per exposure)
# ------------------------------------------------------------------
@dataclass(slots=True)
class SingleStarData:
    subsample: int  # constant for this star
    data: list[np.ndarray] = field(default_factory=list)
    mask: list[np.ndarray] = field(default_factory=list)
    std: list[np.ndarray] = field(default_factory=list)
    psf: list[np.ndarray] = field(default_factory=list)
    star_in_subsampled_pixels: list[tuple[float, float]] = field(default_factory=list)
    observation_ids: list[int] = field(default_factory=list)

    # -- optional: freeze to ndarray stacks -------------------------
    def as_stacked(self) -> "SingleStarDataStacked":
        """Return a read-only view where lists are stacked to 3-D arrays."""
        return SingleStarDataStacked(
            subsample=self.subsample,
            data=np.stack(self.data),
            mask=np.stack(self.mask),
            std=np.stack(self.std),
            psf=np.stack(self.psf),
            star_in_subsampled_pixels=np.asarray(self.star_in_subsampled_pixels),
            observation_ids=np.asarray(self.observation_ids),
        )


@dataclass(slots=True)
class SingleStarDataStacked:
    """Same fields but already stacked into ndarrays."""

    subsample: int
    data: np.ndarray
    mask: np.ndarray
    std: np.ndarray
    psf: np.ndarray
    star_in_subsampled_pixels: np.ndarray
    observation_ids: np.ndarray

    # -- convenience -------------------------------------------------
    @property
    def sky_array(self) -> np.ndarray:
        dshape = self.data.shape
        shape = (dshape[0], dshape[1] * self.subsample, dshape[2] * self.subsample)
        return np.zeros(shape)


# ------------------------------------------------------------------
# 2) Aggregator over many stars
# ------------------------------------------------------------------
class StarData(UserDict[int, SingleStarDataStacked]):
    """
    Merge many StarsBundle instances (different exposures) into
    {star_id -> SingleStarData}.
    """

    def __init__(self, subsample: int, bundles: list[StarsBundle]):
        super().__init__()  # initialise UserDict
        self.subsample = subsample

        # Append every bundle, ordered by exposure index
        for bundle in sorted(bundles, key=lambda b: b.index):
            for star_id, star_bundle in bundle.items():
                self._append_star(star_id, star_bundle)

        # Transform to Stacked Data
        for key, val in self.items():
            self[key] = val.as_stacked()

    # ------------------------------------------------------------------
    def _append_star(self, star_id: int, sb: SingleStarBundle) -> None:
        """Accumulate one exposure’s material for a star."""
        entry = self.setdefault(star_id, SingleStarData(subsample=self.subsample))
        entry.data.append(sb.cutout.data)
        entry.mask.append(sb.cutout.mask)
        entry.std.append(sb.cutout.std)
        entry.psf.append(sb.cutout.psf)
        entry.star_in_subsampled_pixels.append(sb.star_in_subsampled_pixels)
        entry.observation_ids.append(sb.observation_id)
