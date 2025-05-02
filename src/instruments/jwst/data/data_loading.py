from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from astropy.coordinates import SkyCoord

from ..alignment.star_alignment import Star
from ..data.jwst_data import JwstData
from ..parse.alignment.star_alignment import FilterAlignmentMeta


@dataclass
class DataBoundsPreloading:
    shapes: tuple[int, int] | list[tuple[int, int]] | list[np.ndarray] = field(
        default_factory=list
    )
    bounding_indices: (
        tuple[int, int, int, int] | list[tuple[int, int, int, int]] | list[np.ndarray]
    ) = field(default_factory=list)

    def __getitem__(self, index: int) -> "DataBoundsPreloading":
        return DataBoundsPreloading(self.shapes[index], self.bounding_indices[index])

    def align_shapes(self) -> "DataBoundsPreloading":
        shapes = np.array(self.shapes)
        shapes_new = np.full_like(shapes, np.max(shapes, axis=0))

        bounding_indices_new = np.array(self.bounding_indices)
        for ii in range(len(bounding_indices_new)):
            bounding_indices_new[ii, [3, 1]] += shapes_new[ii] - shapes[ii]

        return DataBoundsPreloading(list(shapes_new), list(bounding_indices_new))

    def append_shapes_and_bounds(
        self, jwst_data: JwstData, sky_corners: SkyCoord
    ) -> None:
        self.shapes.append(jwst_data.data_inside_extrema(sky_corners).shape)
        self.bounding_indices.append(
            jwst_data.wcs.bounding_box_indices_from_world_extrema(sky_corners)
        )
