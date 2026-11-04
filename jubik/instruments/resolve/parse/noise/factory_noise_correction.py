"""Parse the `weight_correction` block of a radio data block.

The noise std of each visibility, `1 / sqrt(weight)`, is multiplied by
inferred inverse gamma factors, one per baseline, one per antenna, or both::

    weight_correction:
      baseline: {alpha: 2.0, scale: 1.0}   # sigma_baseline[a, b]
      antenna: {alpha: 2.0, scale: 1.0}    # s[a] * s[b]

With both the factors multiply. Without the block the weights are used as
given.
"""

from dataclasses import dataclass

from .base_line_correction import AntennaCorrection, BaseLineCorrection


@dataclass
class NoiseCorrection:
    baseline: BaseLineCorrection | None
    antenna: AntennaCorrection | None


def factory_noise_correction_parser(
    data_cfg: dict,
) -> NoiseCorrection | None:
    """Parse the noise correction model from `data_cfg["weight_correction"]`,
    see the module docstring. Returns None without the block."""

    model = data_cfg.get("weight_correction", None)

    if model is None:
        return None

    unknown = set(model) - {"baseline", "antenna"}
    if unknown or not model:
        raise ValueError(
            f"weight_correction takes `baseline` and/or `antenna`, got {list(model)}"
        )

    return NoiseCorrection(
        baseline=(
            BaseLineCorrection.from_yaml_dict(model["baseline"])
            if "baseline" in model
            else None
        ),
        antenna=(
            AntennaCorrection.from_yaml_dict(model["antenna"])
            if "antenna" in model
            else None
        ),
    )
