from dataclasses import dataclass

import numpy as np

from .....parse.parsing_base import StaticTyped


@dataclass
class PlottingKwargs(StaticTyped):
    vmin: float | int = -np.inf
    vmax: float | int = np.inf


PLOTTING_KWARGS_DEFAULT = PlottingKwargs()
