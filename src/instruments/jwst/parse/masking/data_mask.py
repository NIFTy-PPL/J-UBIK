from dataclasses import dataclass
from astropy.coordinates import SkyCoord


@dataclass
class CornerMaskConfig:
    corners: list[SkyCoord]

    @classmethod
    def from_yaml_dict(cls, config: dict):
        RA_KEY = "ra"
        DEC_KEY = "dec"

        ras: list[str] = config.get(RA_KEY)
        decs: list[str] = config.get(DEC_KEY)

        # Check consistency
        for name, val in zip([RA_KEY, DEC_KEY], [ras, decs]):
            if not isinstance(val, list) or len(val) != 4:
                raise ValueError(f"{name} should be 4 corners, got: {val}")

        return cls(corners=[SkyCoord(ra=ra, dec=dec) for ra, dec in zip(ras, decs)])


def yaml_to_corner_mask_configs(
    telescope_yaml_dict: dict,
) -> list[CornerMaskConfig]:
    """Factory producing a list of `CornerMaskConfig`

    Parameters
    ----------
    telescope_yaml_dict: dict
        The telescope yaml dict which contains the masks for the corners.
    """

    CORNER_KEY = "corner_mask"

    corner_masks = []
    for key, val in telescope_yaml_dict.items():
        if CORNER_KEY in key.lower():
            corner_masks.append(CornerMaskConfig.from_yaml_dict(val))

    return corner_masks
