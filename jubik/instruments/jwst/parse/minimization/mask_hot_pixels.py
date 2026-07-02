from dataclasses import dataclass


@dataclass
class MaskingStepSettings:
    """Settings for the masking step of JWST data.

    Parameters
    ----------
    mask_at_iteration: int
        The iteration at which to mask.
    threshold_hot_pixel: float
        The threshold value for masking hot pixels. In units of the
        residual = abs(d - Rs) / std.
        If `None`, this masking type is ignored.
    threshold_star_nan: float
        The threshold for the sum of neighboring pixels of a `nan`-pixel. In units of
        np.sqrt((res01**2 + res10**2 + res12**2 + res21**2) / 4).
        If `None`, this masking type is ignored.
    threshold_star_convolution: float
        The threshold for the sum for the residual field convolved by a star pattern.
        If `None`, this masking type is ignored.
    """

    mask_at_iteration: int
    threshold_hot_pixel: float
    threshold_star_nan: float
    threshold_star_convolution: float

    @classmethod
    def from_yaml_dict(cls, raw: dict) -> "MaskingStepSettings":
        masking = raw["masking_step"]
        return cls(
            mask_at_iteration=masking["mask_at_iteration"],
            threshold_hot_pixel=masking.get("threshold_hot_pixel"),
            threshold_star_nan=masking.get("threshold_star_nan"),
            threshold_star_convolution=masking.get("threshold_star_convolution"),
        )
