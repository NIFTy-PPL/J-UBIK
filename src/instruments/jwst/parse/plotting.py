from dataclasses import dataclass, field
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
    def from_yaml_dict(cls, raw: dict):
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
    def from_yaml_dict(cls, raw: dict):
        return cls(
            alpha=FieldPlottingConfig.from_yaml_dict(raw.get("alpha", {})),
            reference=FieldPlottingConfig.from_yaml_dict(raw.get("reference", {})),
            combined=FieldPlottingConfig.from_yaml_dict(raw.get("combined", {})),
        )


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
    residual: FieldPlottingConfig = field(default_factory=FieldPlottingConfig)

    residual_over_std: bool = True
    xmax_residuals: int = 4

    @classmethod
    def from_yaml_dict(cls, raw: dict):
        return cls(
            xmax_residuals=raw.get("xmax_residuals", 4),
            residual_over_std=raw.get("residual_over_std", True),
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
