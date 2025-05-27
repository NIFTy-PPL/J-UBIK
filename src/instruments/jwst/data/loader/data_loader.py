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
)
from .target_loader import (
    TargetBundle,
    TargetData,
    load_one_target_bundle,
)

# --------------------------------------------------------------------------------------
# Load Data Interface
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class DataLoaderTarget:
    """Holds the essential information for loading the data cutouts around the target"""

    grid: Grid
    data_bounds: DataBounds
    subsample: Subsample | int


@dataclass
class DataLoaderStarAlignment:
    """Holds the essential information for loading the data cutouts around the stars"""

    config: StarAlignmentConfig
    tables: StarTables

    @classmethod
    def from_optional(
        cls, config: StarAlignmentConfig | None, tables: StarTables
    ) -> Optional["DataLoaderStarAlignment"]:
        return cls(config, tables) if config else None


@dataclass
class DataLoader:
    """Input data of `load_data`."""

    # Essentials -----------------------------------------------------------------------
    target: DataLoaderTarget
    psf_kernel_configs: JwstPsfKernelConfig

    # Optioanls ------------------------------------------------------------------------
    star_alignment: DataLoaderStarAlignment | None = None
    extra_masks: ExtraMasks | None = None


def load_data(
    filepaths: tuple[IndexAndPath],
    *,
    data_loader: DataLoader,
    loading_mode_config: LoadingModeConfig,
) -> tuple[TargetData, StarData | None]:
    """Load the data
    1. Target cutouts for filter observations.
    2. Psf kernels for the cutouts for filter observations.

    Optional:
    1. Star cutouts.
    2. Psf kernels for the cutouts.

    Parameters
    ----------
    filepaths: tuple[IndexAndPath]
        The filepaths of JWST data files to be preloaded
    data_loader: DataLoader
        - essential:
            target: DataLoaderTarget
                - grid
                - data_bounds
                - subsample
            psf_kernel_config: JwstPsfKernelConfig
                Config parameters for psf kernel loading see `load_psf_kernel`.
        - optional:
            star_alignment: DataLoaderStarAlignment, optional
                - config: StarAlignmentConfig
                - tables: StarTables
            extra_masks: ExtraMasks, optional
                some extra masks, either in the target or the star data cutouts.
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
    logger.info("Loading JWST data")

    target_andor_stars_bundles: list[TargetAndOrStarsBundle] = load_bundles(
        filepaths=filepaths,
        load_one=_load_one_target_and_or_stars,
        extra_kw_args=dict(data_loader=data_loader),
        mode=loading_mode_config.loading_mode,
        workers=loading_mode_config.workers,
    )

    target_data, stars_data = _create_data_products(
        bundles=target_andor_stars_bundles, data_loader=data_loader
    )

    return target_data, stars_data


# --------------------------------------------------------------------------------------
# Load Data Internals
# --------------------------------------------------------------------------------------


@dataclass
class TargetAndOrStarsBundle:
    index: int
    target_bundle: TargetBundle
    stars_bundle: StarsBundle | None


def _load_one_target_and_or_stars(
    filepath: IndexAndPath,
    data_loader: DataLoader,
):
    jwst_data = JwstData(filepath.path)
    target_bundle: TargetBundle = load_one_target_bundle(
        index=filepath.index,
        jwst_data=jwst_data,
        subsample=data_loader.target.subsample,
        target_grid=data_loader.target.grid,
        target_data_bounds=data_loader.target.data_bounds,
        psf_kernel_configs=data_loader.psf_kernel_configs,
        extra_masks=data_loader.extra_masks,
    )

    if data_loader.star_alignment:
        stars_bundle: StarsBundle = load_one_stars_bundle(
            index=filepath.index,
            jwst_data=jwst_data,
            star_tables=data_loader.star_alignment.tables,
            star_alignment_config=data_loader.star_alignment.config,
            extra_masks=data_loader.extra_masks,
            psf_kernel_configs=data_loader.psf_kernel_configs,
        )
    else:
        stars_bundle = None

    return TargetAndOrStarsBundle(
        index=filepath.index,
        target_bundle=target_bundle,
        stars_bundle=stars_bundle,
    )


def _create_data_products(
    bundles: list[TargetAndOrStarsBundle], data_loader: DataLoader
) -> tuple[TargetData, StarData | None]:
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

    target_bundles = []
    stars_bundles = []
    for b in bundles:
        target_bundles.append(b.target_bundle)
        if b.stars_bundle is not None:
            stars_bundles.append(b.stars_bundle)

    target_data = TargetData.from_bundles(data_loader.target.subsample, target_bundles)

    stars_data = (
        None
        if stars_bundles == []
        else StarData(data_loader.star_alignment.config.subsample, stars_bundles)
    )

    return target_data, stars_data
