from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class HotPixelMaskingData:
    filter: list[str] = field(default_factory=list)
    nan_mask: list[NDArray] = field(default_factory=list)
    bounds: list[NDArray] = field(default_factory=list)

    def append_information(
        self,
        filter: str,
        nan_mask: NDArray,
        bounds: list[tuple[int, int, int, int]] | list[NDArray],
    ):
        self.filter.append(filter)
        self.nan_mask.append(nan_mask)
        self.bounds.append(np.array(bounds))

    def get_filter_nanmask(self, filter_name: str):
        return self.nan_mask[self.filter.index(filter_name)]

    def get_filter_bounds(self, filter_name: str):
        return self.bounds[self.filter.index(filter_name)]
