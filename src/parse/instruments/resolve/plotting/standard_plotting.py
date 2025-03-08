from ....parsing_base import StronglyTyped

import numpy as np

from dataclasses import dataclass


@dataclass
class PlottingKwargs(StronglyTyped):
    vmin: float | int = -np.inf
    vmax: float | int = np.inf


PLOTTING_KWARGS_DEFAULT = PlottingKwargs()
