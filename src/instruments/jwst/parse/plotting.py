from dataclasses import dataclass, field
from typing import Optional

import nifty8.re as jft
import numpy as np


@dataclass
class FieldPlottingConfig:
    vmin: float | None = None
    vmax: float | None = None
    norm: str | None = None
    cmap: str = "viridis"

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

    @classmethod
    def from_yaml_dict(cls, raw: dict) -> "FieldPlottingConfig":
        return cls(
            vmin=raw.get("vmin"),
            vmax=raw.get("vmax"),
            norm=raw.get("norm"),
            cmap=raw.get("cmap", "viridis"),
        )


@dataclass
class MultiFrequencyPlottingConfig:
    alpha: FieldPlottingConfig = field(default_factory=FieldPlottingConfig)
    reference: FieldPlottingConfig = field(default_factory=FieldPlottingConfig)
    combined: FieldPlottingConfig = field(default_factory=FieldPlottingConfig)

    @classmethod
    def from_yaml_dict(cls, raw: dict) -> "MultiFrequencyPlottingConfig":
        return cls(
            alpha=FieldPlottingConfig.from_yaml_dict(raw.get("alpha", {})),
            reference=FieldPlottingConfig.from_yaml_dict(raw.get("reference", {})),
            combined=FieldPlottingConfig.from_yaml_dict(raw.get("combined", {})),
        )


@dataclass
class LensSystemPlottingConfig:
    share_source_vmin_vmax: bool = False  # Sharing vmin, vmax for the source brightness
    rgb_plotting: bool = False
    source: MultiFrequencyPlottingConfig = field(
        default_factory=MultiFrequencyPlottingConfig
    )
    lens_mass: FieldPlottingConfig = field(default_factory=FieldPlottingConfig)
    lens_light: MultiFrequencyPlottingConfig = field(
        default_factory=MultiFrequencyPlottingConfig
    )

    @classmethod
    def from_yaml_dict(cls, raw: dict) -> "LensSystemPlottingConfig":
        return cls(
            share_source_vmin_vmax=raw.get("share_source_vmin_vmax", False),
            rgb_plotting=raw.get("rgb_plotting", False),
            source=MultiFrequencyPlottingConfig.from_yaml_dict(raw.get("source")),
            lens_mass=FieldPlottingConfig.from_yaml_dict(raw.get("mass", {})),
            lens_light=MultiFrequencyPlottingConfig.from_yaml_dict(
                raw.get("lens_light")
            ),
        )


@dataclass
class ResidualOverplot:
    max_percent_contours: list[float] | None
    contour_settings: dict = field(default_factory=dict)
    overplot_model: jft.Model | None = None

    @classmethod
    def from_optional(cls, raw: dict | None) -> Optional["ResidualOverplot"]:
        if raw is None:
            return None

        return cls(
            max_percent_contours=raw.get("max_percent_contours"),
            contour_settings=raw.get("contour_settings", {}),
        )


@dataclass
class ResidualPlottingConfig:
    sky: FieldPlottingConfig = field(default_factory=FieldPlottingConfig)
    data: FieldPlottingConfig = field(default_factory=FieldPlottingConfig)
    residual: FieldPlottingConfig = field(default_factory=FieldPlottingConfig)

    residual_over_std: bool = True
    xmax_residuals: int = 4
    residual_overplot: ResidualOverplot | None = None

    @classmethod
    def from_yaml_dict(cls, raw: dict) -> "ResidualPlottingConfig":
        return cls(
            xmax_residuals=raw.get("xmax_residuals", 4),
            residual_over_std=raw.get("residual_over_std", True),
            residual_overplot=ResidualOverplot.from_optional(
                raw.get("residual_overplot")
            ),
            sky=FieldPlottingConfig.from_yaml_dict(raw.get("sky", {})),
            data=FieldPlottingConfig.from_yaml_dict(raw.get("data", {})),
            residual=FieldPlottingConfig.from_yaml_dict(
                raw.get("residual", dict(vmin=-3, vmax=3, norm=None, cmap="RdBu_r"))
            ),
        )

    def __post_init__(self):
        if self.residual == FieldPlottingConfig():
            self.residual = FieldPlottingConfig(
                vmin=-3, vmax=3, norm=None, cmap="RdBu_r"
            )
