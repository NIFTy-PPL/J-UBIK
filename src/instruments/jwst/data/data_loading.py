from dataclasses import dataclass, field
import numpy as np


@dataclass
class DataBoundsPreloading:
    shapes: tuple[int, int] | list[tuple[int, int]] | list[np.ndarray] = field(
        default_factory=list
    )
    bounding_indices: (
        tuple[int, int, int, int] | list[tuple[int, int, int, int]] | list[np.ndarray]
    ) = field(default_factory=list)

    def __getitem__(self, index: int):
        return DataBoundsPreloading(self.shapes[index], self.bounding_indices[index])

    def align_shapes(self):
        shapes = np.array(self.shapes)
        shapes_new = np.full_like(shapes, np.max(shapes, axis=0))

        bounding_indices_new = np.array(self.bounding_indices)
        for ii in range(len(bounding_indices_new)):
            bounding_indices_new[ii, [3, 1]] += shapes_new[ii] - shapes[ii]

        return DataBoundsPreloading(list(shapes_new), list(bounding_indices_new))
