from dataclasses import dataclass, field
import numpy as np


@dataclass
class Preloading:
    shapes: tuple[int, int] | list[tuple[int, int]] | list[np.ndarray] = field(
        default_factory=list
    )
    bounding_indices: (
        tuple[int, int, int, int] | list[tuple[int, int, int, int]] | list[np.ndarray]
    ) = field(default_factory=list)

    def append_shape(self, shape: tuple[int, int]):
        self.shapes.append(shape)

    def append_bounding_indices(self, bounding_indices: tuple[int, int, int, int]):
        self.bounding_indices.append(bounding_indices)

    def __getitem__(self, index: int):
        return Preloading(self.shapes[index], self.bounding_indices[index])

    def align_data(self):
        shapes = np.array(self.shapes)
        shapes_new = np.full_like(shapes, np.max(shapes, axis=0))

        bounding_indices_new = np.array(self.bounding_indices)
        for ii in range(len(bounding_indices_new)):
            bounding_indices_new[ii, [3, 1]] += shapes_new[ii] - shapes[ii]

        return Preloading(list(shapes_new), list(bounding_indices_new))
