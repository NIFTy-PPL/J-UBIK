from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


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

    bounds: list[NDArray] = field(default_factory=list)

    def align_bounds(self) -> "DataBounds":
        """Alignt the data shapes, such that all cutouts have the same shape. The bounds
        for the cutout are adjusted, accordingly."""
        shapes_new = np.full_like(self.shapes, np.max(self.shapes, axis=0))

        bounding_indices_new = np.array(self.bounds)
        for ii in range(len(bounding_indices_new)):
            bounding_indices_new[ii, [1, 3]] += shapes_new[ii] - self.shapes[ii]

        return DataBounds(list(bounding_indices_new))

    def add_cutout(self, bounds: tuple[int, int, int, int]) -> None:
        """Append the shape and bounding indices for a data cutout."""
        assert isinstance(self.bounds, list), "Can only add cutout when bound is a list"
        self.bounds.append(np.array(bounds))

    @property
    def shapes(self):
        bnd = np.array(self.bounds)
        return np.array((bnd[:, 1] - bnd[:, 0], bnd[:, 3] - bnd[:, 2])).T

    def cast_to_numpy(self):
        self.bounds = np.array(self.bounds)

    def adjust_bounds(self, adjust: NDArray):
        self.bounds = self.bounds + adjust
