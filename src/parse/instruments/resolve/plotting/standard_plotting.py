from ....parsing_base import StaticTyped

import numpy as np

from dataclasses import dataclass


@dataclass
class PlottingKwargs(StaticTyped):
    vmin: float | int = -np.inf
    vmax: float | int = np.inf


PLOTTING_KWARGS_DEFAULT = PlottingKwargs()
