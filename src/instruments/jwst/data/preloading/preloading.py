import time
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Callable, Literal

import numpy as np
from astropy.coordinates import SkyCoord
from nifty8.re import logger

from .....color import Color
from ....gaia.star_finder import load_gaia_stars_in_fov
from ...alignment.filter_alignment import FilterAlignment
from ..concurrent_loading import load_bundles
from ..jwst_data import JwstData
from .checks import PreloadingChecks
from .data_bounds import DataBounds
from ...parse.data.data_loading import LoadingMode


@dataclass(slots=True)
class PreloadBundle:
    index: int
    filepath: str
    pivot_color: Color
    bounding_indices: tuple[int, int, int, int]
    boresight: SkyCoord
    jwst_world_corners: SkyCoord


def load_one_preload(index: int, filepath: str, grid_corners: SkyCoord):
    jwst_data = JwstData(filepath)

    return PreloadBundle(
        index=index,
        filepath=filepath,
        pivot_color=jwst_data.pivot_wavelength,
        bounding_indices=jwst_data.wcs.bounding_indices_from_world_extrema(
            grid_corners
        ),
        boresight=jwst_data.get_boresight_world_coords(),
        jwst_world_corners=jwst_data.wcs.world_corners(),
    )


def _apply_preload_side_effects(
    bundles: Iterable[PreloadBundle],
    target_bounds: DataBounds,
    checks: PreloadingChecks,
    filter_alignment: FilterAlignment,
) -> None:
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
    target_bounds: DataBounds
        The target bounds object to update with cutout indices
    checks: PreloadingChecks
        Object containing validation checks to perform on the data
    filter_alignment: FilterAlignment
        The alignment data to update with:
            - boresight information
            - star positions (if star_alignment is enabled)

    Returns
    -------
    None
    """
    for b in bundles:
        logger.info(f"Preload: {b.index} {b.filepath}")
        checks.check_energy_consistency(b.pivot_color, b.filepath)
        target_bounds.add_cutout(b.bounding_indices)

        filter_alignment.boresight.append(b.boresight)
        if filter_alignment.star_alignment:
            filter_alignment.star_alignment.star_tables.append(
                load_gaia_stars_in_fov(
                    b.jwst_world_corners,
                    filter_alignment.star_alignment.alignment_meta.library_path,
                )
            )


def data_preloading(
    filepaths: tuple[Path],
    grid_corners: list[SkyCoord],
    filter_alignment: FilterAlignment,
    loading_mode: LoadingMode,
    workers: int | None = None,
) -> tuple[Color, DataBounds, FilterAlignment]:
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
    filepaths: tuple[Path]
        The filepaths of JWST data files to be preloaded
    grid_corners: list[SkyCoord]
        The spatial corners defining the sky grid
    filter_alignment: FilterAlignment
        Container for alignment data, updated with:
        - boresight coordinates
        - star positions (when star_alignment is enabled)
    loading_mode: Literal["processes", "threads", "serial"] = "serial"
        Method for loading data:
        - "serial": Sequential processing
        - "threads": Multi-threaded for I/O-bound operations
        - "processes": Multi-process for CPU-bound operations
    workers: int | None
        Number of threads/processes to use (None = executor default)

    Returns
    -------
    Tuple[Color, DataBounds, FilterAlignment]
        - Color: Detected color/filter information
        - DataBounds: Aligned shapes and bounds for the target data
        - FilterAlignment: Updated alignment information
    """

    t = time.perf_counter()
    logger.info("Preloading JWST data")

    target_bounds = DataBounds()
    preloading_checks = PreloadingChecks()

    bundles: Iterable[PreloadBundle] = load_bundles(
        filepaths,
        load_one_preload,
        mode=loading_mode,
        workers=workers,
        extra_pos_args=(grid_corners,),
    )

    _apply_preload_side_effects(
        bundles, target_bounds, preloading_checks, filter_alignment
    )
    target_bounds = target_bounds.align_shapes_and_bounds()

    logger.info(f"{time.perf_counter() - t}")

    return preloading_checks.color, target_bounds, filter_alignment
