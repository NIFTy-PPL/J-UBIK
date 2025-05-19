from dataclasses import dataclass
from typing import Optional

from nifty8.re import logger

from .....grid import Grid
from ...alignment.star_alignment import StarTables
from ...parse.alignment.star_alignment import StarAlignmentConfig
from ...parse.data.data_loader import IndexAndPath, LoadingModeConfig, Subsample
from ...parse.jwst_psf import JwstPsfKernelConfig
from ...parse.masking.data_mask import ExtraMasks
from ..concurrent_loader import load_bundles
from ..jwst_data import JwstData
from ..preloader.data_bounds import DataBounds
from .stars_loader import (
    StarData,
    StarsBundle,
    load_one_stars_bundle,
    load_one_stars_bundle_from_filepath,
)
from .target_loader import (
    TargetBundle,
    TargetData,
    load_one_target_bundle,
    load_one_target_bundle_from_filepath,
)


@dataclass
class DataLoaderTarget:
    grid: Grid
    data_bounds: DataBounds
    subsample: Subsample | int


@dataclass
class DataLoaderStarAlignment:
    config: StarAlignmentConfig
    tables: StarTables

    @classmethod
    def from_optional(
        cls, config: StarAlignmentConfig | None, tables: StarTables
    ) -> Optional["DataLoaderStarAlignment"]:
        return cls(config, tables) if config else None


def load_data(
    filepaths: tuple[IndexAndPath],
    target: DataLoaderTarget,
    star_alignment: DataLoaderStarAlignment | None,
    psf_kernel_configs: JwstPsfKernelConfig,
    extra_masks: ExtraMasks,
    loading_mode_config: LoadingModeConfig,
) -> tuple[TargetData, StarData]:
    target_bundles: list[TargetBundle] = load_bundles(
        filepaths=filepaths,
        load_one=load_one_target_bundle_from_filepath,
        extra_kw_args=dict(
            subsample=target.subsample,
            target_grid=target.grid,
            target_data_bounds=target.data_bounds,
            psf_kernel_configs=psf_kernel_configs,
            extra_masks=extra_masks,
        ),
        mode=loading_mode_config.loading_mode,
        workers=loading_mode_config.workers,
    )

    stars_bundles: list[StarsBundle] = load_bundles(
        filepaths=filepaths,
        load_one=load_one_stars_bundle_from_filepath,
        extra_kw_args=dict(
            star_tables=star_alignment.tables,
            star_alignment_config=star_alignment.config,
            extra_masks=extra_masks,
            psf_kernel_configs=psf_kernel_configs,
        ),
        mode=loading_mode_config.loading_mode,
        workers=loading_mode_config.workers,
    )

    return (
        TargetData.from_bundles(target.subsample, target_bundles),
        StarData(star_alignment.config.subsample, stars_bundles),
    )
