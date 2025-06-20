from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class DataCutout:
    data: np.ndarray[float]
    mask: np.ndarray[bool]
    std: np.ndarray[float]
    psf: np.ndarray[float]
    nan_mask: np.ndarray[float]
