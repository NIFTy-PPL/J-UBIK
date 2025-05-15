from nifty8.re import logger

from .....grid import Grid
from .....color import Color
from ...alignment.filter_alignment import FilterAlignment
from ...parse.data.data_loading import FilterAndFilePaths
from ...parse.jwst_response import SkyMetaInformation
from ..jwst_data import JwstData
from .checks import PreloadingChecks
from .star_alignment import star_alignment_preloading
from .target import DataBounds, target_preloading


def data_preloading(
    filter_and_filepaths: FilterAndFilePaths,
    grid: Grid,
    sky_meta: SkyMetaInformation,
    filter_alignment: FilterAlignment,
) -> tuple[Color, DataBounds, FilterAlignment]:
    """Preloading step on the data, which performs:
    1. Preloading checks
        - check_energy_consistency
    2. Preloading target
        - align the shapes for the target data.
    3. Preloding star alignment, optional
        - load the stars in the field

    Parameters
    ----------

    Returns
    -------
    (color_of_filter, target_bounds, filter_alignment)

    """
    logger.info("Preloading JWST data")

    target_bounds = DataBounds()
    preloading_checks = PreloadingChecks()

    for ii, filepath in enumerate(filter_and_filepaths.filepaths):
        print(ii, filepath)
        jwst_data = JwstData(filepath)
        preloading_checks.check_energy_consistency(jwst_data.pivot_wavelength, filepath)

        target_preloading(
            target_bounds,
            jwst_data,
            grid,
            sky_meta,
        )
        if filter_alignment.star_alignment is not None:
            star_alignment_preloading(filter_alignment.star_alignment, jwst_data)

    target_bounds = target_bounds.align_shapes_and_bounds()

    return preloading_checks.color, target_bounds, filter_alignment
