from dataclasses import dataclass
from collections import UserList
from astropy.coordinates import SkyCoord


@dataclass
class ExtraMaskFromCorners:
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


class ExtraMasks(UserList):
    """A list of `ExtraMaskFromCorners`."""

    @classmethod
    def from_yaml_dict(cls, raw: dict | None):
        """Factory producing a ExtraMasks object with multiple `ExtraMaskFromCorners`

        Parameters
        ----------
        raw: dict, parsed yaml file
        - `corner_mask`, optional
            If the string `corner_mask` is inside the yaml dict, the mask will be
            appended to the list.
        """

        CORNER_KEY = "corner_mask"

        corner_masks = []
        for key, val in raw.items():
            if CORNER_KEY in key.lower():
                corner_masks.append(ExtraMaskFromCorners.from_yaml_dict(val))

        return cls(corner_masks)
