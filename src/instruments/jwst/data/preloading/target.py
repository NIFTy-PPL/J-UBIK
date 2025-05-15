from dataclasses import dataclass, field

import numpy as np

from .....grid import Grid
from ...parse.jwst_response import SkyMetaInformation
from ..jwst_data import JwstData


@dataclass
class DataBounds:
    """This class holds the shapes and bounds (min-max column, min-max row) for a data
    cutout. Intended use-case is to align the shapes and sizes of a data cutout around
    the target.

    Parameters
    ----------
    shapes: list[tuple[int, int]], list[np.ndarray]
        The shape of the cutout.
    bounds: list[tuple[int, int, int int]], list[np.ndarray]
        The bounds of the data cutout inside the target data.
    """

    shapes: list[tuple[int, int]] | list[np.ndarray] = field(default_factory=list)
    bounds: list[tuple[int, int, int, int]] | list[np.ndarray] = field(
        default_factory=list
    )

    def align_shapes_and_bounds(self) -> "DataBounds":
        """Alignt the data shapes, such that all cutouts have the same shape. The bounds
        for the cutout are adjusted, accordingly."""
        shapes = np.array(self.shapes)
        shapes_new = np.full_like(shapes, np.max(shapes, axis=0))

        bounding_indices_new = np.array(self.bounds)
        for ii in range(len(bounding_indices_new)):
            bounding_indices_new[ii, [1, 3]] += shapes_new[ii] - shapes[ii]

        return DataBounds(list(shapes_new), list(bounding_indices_new))

    def append_shapes_and_bounds(self, bounds: tuple[int, int, int, int]) -> None:
        """Append the shape and bounding indices for a data cutout."""
        shape = bounds[1] - bounds[0], bounds[3] - bounds[2]
        self.shapes.append(shape)
        self.bounds.append(bounds)


def target_preloading(
    target_bounds: DataBounds,
    jwst_data: JwstData,
    grid: Grid,
    sky_meta: SkyMetaInformation,
):
    """Preloading step on the data. This performs:
    1. Appending the the bounds and shapes of the target grid in the data.

    Parameters
    ----------

    """

    bounding_indices = jwst_data.wcs.bounding_indices_from_world_extrema(
        grid.spatial.world_corners(extension_value=sky_meta.grid_extension)
    )
    target_bounds.append_shapes_and_bounds(bounds=bounding_indices)
