import time
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from nifty8.re import logger

from .....color import Color
from ....gaia.star_finder import load_gaia_stars_in_fov
from ...alignment.filter_alignment import FilterAlignment
from ...alignment.star_alignment import StarTables
from ...parse.alignment.star_alignment import StarAlignmentConfig
from ...parse.data.data_loader import IndexAndPath, LoadingModeConfig
from ..concurrent_loader import load_bundles
from ..jwst_data import DataMetaInformation, JwstData
from .checks import FilterConsistency
from .data_bounds import DataBounds


@dataclass(slots=True)
class PreloadBundle:
    index: int
    filepath: IndexAndPath
    pivot_color: Color
    bounding_indices: tuple[int, int, int, int]
    boresight: SkyCoord
    meta: DataMetaInformation
    star_table: Table | None
    date: str


def _load_one_preload_bundle(
    filepath: IndexAndPath,
    grid_corners: SkyCoord,
    star_alignment_config: StarAlignmentConfig | None,
):
    jwst_data = JwstData(filepath.path)

    if star_alignment_config:
        jwst_world_corners = jwst_data.wcs.world_corners()
        star_table = load_gaia_stars_in_fov(
            fov_corners=jwst_world_corners,
            library_path=star_alignment_config.library_path,
            exclude_source_ids=star_alignment_config.exclude_source_id,
        )
    else:
        star_table = None

    return PreloadBundle(
        index=filepath.index,
        filepath=filepath,
        pivot_color=jwst_data.pivot_wavelength,
        bounding_indices=jwst_data.wcs.bounding_indices_from_world_extrema(
            grid_corners
        ),
        boresight=jwst_data.get_boresight_world_coords(),
        star_table=star_table,
        meta=jwst_data.meta,
        date=jwst_data.dm.meta.date,
    )


def _preload_data_products(
    bundles: Iterable[PreloadBundle],
    star_alignment_config: StarAlignmentConfig | None,
) -> tuple[DataMetaInformation, DataBounds, StarTables | None, list[SkyCoord]]:
    """Apply side effects from the preloaded data bundles.

    This function processes each preload bundle to:
    1. Log preload information
    2. Perform energy consistency checks
    3. Update target bounds with cutout data
    4. Update filter alignment with boresight and optional star data

    Parameters
    ----------
    bundles: Iterable[PreloadBundle]
        The preloaded data bundles to process
    star_alignment_config: StarAlignmentMeta
        The Star alignment meta information, needed for instantiating the StarTables.

    Returns
    -------
    DataMetaInformation
    DataBounds
    StarTables
    boresights: list[SkyCoord]
    """
    checks = FilterConsistency()
    target_bounds = DataBounds()
    star_tables = StarTables() if star_alignment_config else None
    boresights: list[SkyCoord] = []

    for b in bundles:
        logger.info(f"Preload: {b.index} {b.filepath.path}")
        checks.check_meta_consistency(b.meta, b.filepath.path)

        target_bounds.add_cutout(b.bounding_indices)

        boresights.append(b.boresight)
        if star_tables is not None:
            star_tables.append(b.star_table, b.date)

    target_bounds = target_bounds.align_shapes_and_bounds()
    return checks.meta, target_bounds, star_tables, boresights


def _preload_side_effects(
    filter_alignment: FilterAlignment,
    boresights: list[SkyCoord],
) -> None:
    """Apply side effects from the preloaded data products."""
    filter_alignment.boresight = boresights


@dataclass
class PreloaderEssentials:
    grid_corners: list[SkyCoord]


@dataclass
class PreloaderOptionals:
    star_alignment_config: StarAlignmentConfig | None

    @classmethod
    def from_optional(
        cls,
        star_alignment_config: StarAlignmentConfig | None,
    ) -> Optional["PreloaderOptionals"]:
        return cls(
            star_alignment_config=star_alignment_config,
        )


@dataclass
class PreloaderSideEffects:
    filter_alignment: FilterAlignment


def preload_data(
    filepaths: tuple[IndexAndPath],
    essential: PreloaderEssentials,
    optional: PreloaderOptionals,
    side_effects: PreloaderSideEffects,
    loading_mode_config: LoadingModeConfig,
) -> tuple[DataMetaInformation, DataBounds, StarTables | None]:
    """Preload JWST data and perform validation checks.

    This function processes filepaths to:
    #### 1. Perform validation checks
    - Verify energy consistency across files

    #### 2. Build target data information
    - Load and align shapes for target data
    - Collect corresponding cutout indices

    #### 3. Gather alignment information
    - Record boresight coordinates
    - Optionally load stars in the field of view for alignment

    Parameters
    ----------
    filepaths: tuple[IndexAndPath]
        The filepaths of JWST data files to be preloaded
    essential: PreloaderEssentials:
        grid_corners: list[SkyCoord], The spatial corners defining the sky grid
    optional: PreloaderOptionals,
        star_alignment_config, triggers a star search in the gaia catalog
            - library_path (where to save the tables)
            - exclude_source_id (which sources to exclude from the table)
    side_effects:
        filter_alignment: FilterAlignment
            Container for alignment data, updated with:
            - boresight coordinates
    loading_mode_config: LoadingModeConfig:
        loading_mode: LoadingMode, algorithm for loading data:
            - "serial": Sequential processing
            - "threads": Multi-threaded for I/O-bound operations
            - "processes": Multi-process for CPU-bound operations
        workers: int | None
            Number of threads/processes to use (None = executor default)

    Returns
    -------
    Tuple[DataMetaInformation, DataBounds, StarTables | None]
        - DataMetaInformation: Filter information, checked for consistency.
        - DataBounds: Aligned shapes and bounds for the target data
        - StarTables, optional: Star tables from the gaia catalog
    """

    t = time.perf_counter()
    logger.info("Preload JWST data")

    bundles: Iterable[PreloadBundle] = load_bundles(
        filepaths,
        _load_one_preload_bundle,
        mode=loading_mode_config.loading_mode,
        workers=loading_mode_config.workers,
        extra_kw_args=dict(
            grid_corners=essential.grid_corners,
            star_alignment_config=optional.star_alignment_config,
        ),
    )

    filter_meta, target_bounds, star_tables, boresights = _preload_data_products(
        bundles,
        optional.star_alignment_config,
    )

    _preload_side_effects(side_effects.filter_alignment, boresights)

    logger.info(f"{time.perf_counter() - t}")

    return filter_meta, target_bounds, star_tables
