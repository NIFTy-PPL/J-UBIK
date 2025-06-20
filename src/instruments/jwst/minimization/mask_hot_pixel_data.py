from dataclasses import dataclass, field

import numpy as np


@dataclass
class HotPixelMaskingData:
    filter: list[str] = field(default_factory=list)
    nan_mask: list[np.ndarray] = field(default_factory=list)

    def append_information(
        self,
        filter: str,
        nan_mask: np.ndarray,
    ):
        self.filter.append(filter)
        self.nan_mask.append(nan_mask)
