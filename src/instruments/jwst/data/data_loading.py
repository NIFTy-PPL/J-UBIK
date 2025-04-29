from dataclasses import dataclass, field


@dataclass
class Preloading:
    shapes: tuple[int, int] | list[tuple[int, int]] = field(default_factory=list)
    bounding_indices: tuple[int, int, int, int] | list[tuple[int, int, int, int]] = (
        field(default_factory=list)
    )

    def append_shape(self, shape: tuple[int, int]):
        self.shapes.append(shape)

    def append_bounding_indices(self, bounding_indices: tuple[int, int, int, int]):
        self.bounding_indices.append(bounding_indices)

    def __getitem__(self, index: int):
        return Preloading(self.shapes[index], self.bounding_indices[index])

