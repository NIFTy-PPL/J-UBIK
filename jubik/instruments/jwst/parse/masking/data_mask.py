import hashlib
from collections import UserDict, UserList
from dataclasses import dataclass
from os.path import isfile

import numpy as np
from astropy.coordinates import SkyCoord
from numpy.typing import NDArray

# Longest bounds part kept as written: with a prefix of up to 21 characters
# (``extra_mask_f2100w_b`` has 19) and ``.npy`` a mask file name has at most
# 255 bytes, the file-name limit.
MAX_BOUNDS_NAME = 230


@dataclass
class ExtraMaskFromCorners:
    corners: list[SkyCoord]
    filters: list[str] | None = None

    def __call__(
            self,
            filter_name
        ) -> "ExtraMaskFromCorners | None":
        if isinstance(self.filters, list):
            if filter_name not in self.filters:
                return None
        return self

    @classmethod
    def from_yaml_dict(cls, config: dict):
        RA_KEY = "ra"
        DEC_KEY = "dec"
        FILTER_KEY = "filters"

        ras: list[str] = config.get(RA_KEY)
        decs: list[str] = config.get(DEC_KEY)
        filters: list[str] = config.get(FILTER_KEY, None)

        # Check consistency
        for name, val in zip([RA_KEY, DEC_KEY], [ras, decs]):
            if not isinstance(val, list) or len(val) != 4:
                raise ValueError(f"{name} should be 4 corners, got: {val}")

        return cls(
            corners=[SkyCoord(ra=ra, dec=dec) for ra, dec in zip(ras, decs)],
            filters=filters
        )


class CornerMasks(UserList):
    """A list of `ExtraMaskFromCorners`."""

    def __call__(
        self,
        filter_name: str
    ) -> "CornerMasks | None":
        res = []
        for mask in self:
            if mask(filter_name) is not None:
                res.append(mask)
        return CornerMasks(res) if res else None

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


@dataclass
class NanMaskLoader:
    path: str

    def load_mask(self):
        return np.load(self.path)

    def __post_init__(self):
        if not isfile(self.path):
            raise IOError(f"Loading nan mask: {self.path} doesn't exist")


class NanMasks:
    """A wrapper for multiple NanMaskLoader with the the filter-name as a key."""

    def __init__(self, map: dict) -> None:
        self.map: dict = map

    @classmethod
    def from_yaml_dict(cls, raw: dict):
        """Factory producing a ExtraMasks object with multiple `ExtraMaskFromCorners`

        Parameters
        ----------
        raw: dict, parsed yaml file
        - `nan_mask`, optional
            If the string `nan_mask` is inside the yaml dict, we get the filter and path
        """
        nan_mask_dict = raw.get("nan_mask", {})
        map = {
            filter_name: NanMaskLoader(path)
            for filter_name, path in nan_mask_dict.items()
        }

        return cls(map)

    @staticmethod
    def get_bounds_name(bounds: NDArray):
        """The bounds part of a mask file name ``<prefix>b<bounds name>.npy``.

        Every exposure's cutout bounds, joined; past ``MAX_BOUNDS_NAME``
        characters (12 exposures of four-digit bounds, 14 of three-digit ones)
        the SHA-256 of that string in decimal, so the file name stays within
        the 255-byte limit. The digest is the
        same on every machine and has no ``b``, ``/`` or ``.`` for `get` to
        split on.
        """
        name = "__".join(["_".join([str(b) for b in bound]) for bound in bounds])
        if len(name) <= MAX_BOUNDS_NAME:
            return name
        return str(int(hashlib.sha256(name.encode()).hexdigest(), 16))

    def get(
        self,
        key: str,
        bounds: list[tuple[int, int, int, int]] | list[NDArray],
    ) -> None | NanMaskLoader:
        val = self.map.get(key)

        if val is None:
            return None

        name = self.get_bounds_name(bounds)
        bound_name = val.path.split("/")[-1].split("b")[-1].split(".")[0]

        if name != bound_name:
            raise ValueError(
                f"{key} bounds and extra nan-mask do not match. The cutout follows "
                "the sky grid and target.data_arcsec_extension, which unless set is "
                "automatic and follows the shift and rotation priors too: rebuild "
                "the mask for these bounds, set a number to keep them fixed, or set "
                "'off' for the PSF-sized cutout.\n"
                f"load='{name}'\n"
                f"mask='{bound_name}'\n"
                f"bounds={np.asarray(bounds).tolist()}\n"
            )

        return val
