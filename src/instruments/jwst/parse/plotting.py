from dataclasses import dataclass, field
import numpy as np


@dataclass
class FieldPlottingConfig:
    vmin: float | None = None
    vmax: float | None = None
    norm: str | None = None

    # rendering: dict = field(default_factory=dict(origin="lower", interpolation="None"))

    def get_min(self, field: np.ndarray) -> float:
        f"""Returns maximum between field.max and {self.vmin}"""
        min = -np.inf if self.vmin is None else self.vmin
        return np.max((np.nanmin(field), min))

    def get_max(self, field: np.ndarray) -> float:
        f"""Returns minimum between field max and {self.vmax}"""
        max = np.inf if self.vmax is None else self.vmax
        return np.min((np.nanmax(field), max))

    @property
    def rendering(self):
        return dict(origin="lower", interpolation="None")


@dataclass
class MultiFrequencyPlottingConfig:
    alpha: FieldPlottingConfig = field(default_factory=FieldPlottingConfig)
    reference: FieldPlottingConfig = field(default_factory=FieldPlottingConfig)
    combined: FieldPlottingConfig = field(default_factory=FieldPlottingConfig)


@dataclass
class LensSystemPlottingConfig:
    share_source_vmin_vmax: bool = False  # Sharing vmin, vmax for the source brightness
    source: MultiFrequencyPlottingConfig = field(
        default_factory=MultiFrequencyPlottingConfig
    )
    lens_light: MultiFrequencyPlottingConfig = field(
        default_factory=MultiFrequencyPlottingConfig
    )
    lens_mass: FieldPlottingConfig = field(default_factory=FieldPlottingConfig)


@dataclass
class ResidualPlottingConfig:
    sky: FieldPlottingConfig = field(default_factory=FieldPlottingConfig)
    data: FieldPlottingConfig = field(default_factory=FieldPlottingConfig)

    std_relative: bool = True
    display_pointing: bool = True
    display_chi2: bool = True
    fileformat: str = "png"
    xmax_residuals: int = np.inf
    ylen_offset: int = 0
