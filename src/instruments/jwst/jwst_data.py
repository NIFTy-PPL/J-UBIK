# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %
from dataclasses import dataclass

import numpy as np
from astropy import units
from astropy.coordinates import SkyCoord
from numpy.typing import ArrayLike

from ...color import Color, ColorRange
from ...grid import Grid
from ...wcs.wcs_jwst_data import WcsJwstData

from .jwst_information import get_dvol, JWST_FILTERS
from .masking import (
    get_mask_from_index_centers_within_rgrid,
    get_mask_from_mask_corners,
)
from .parse.masking.data_mask import CornerMaskConfig

try:
    from jwst import datamodels
except ImportError:
    print("jwst not installed. Some JWST functions will not work.")
    pass


@dataclass
class DataMetaInformation:
    identifier: str
    subsample: int
    unit: units.Unit
    dvol: units.Quantity


class JwstData:
    """Class to contain JWST data metadata."""

    def __init__(self, filepath: str, identifier: str = "", subsample: int = 1):
        """
        Initializes the JwstData class.

        Parameters
        ----------
        filepath : str
            Path to the JWST data file.
        """
        self.dm = datamodels.open(filepath)
        self.wcs = WcsJwstData(self.dm.meta.wcs)
        self.shape = self.dm.data.shape
        self.filter = self.dm.meta.instrument.filter.upper()
        self.camera = self.dm.meta.instrument.name.upper()

        self.meta: DataMetaInformation = DataMetaInformation(
            identifier=identifier,
            subsample=subsample,
            unit=units.Unit(self.dm.meta.bunit_data),
            dvol=get_dvol(self.filter),
        )

    def data_inside_extrema(self, extrema: SkyCoord) -> ArrayLike:
        """
        Find the data values inside the extrema.

        Parameters
        ----------
        extrema : List[SkyCoord]
            List of SkyCoord objects, representing the world location of the
            reconstruction grid edges.

        Returns
        -------
        data : ArrayLike
            Data values inside the extrema.
        """
        # NOTE : Carefull replacing bounding_box_indices_from_world_extrema since this
        # is coupled inside self.load_cutout_data_mask_std_by_world_corners.

        minx, maxx, miny, maxy = self.wcs.bounding_box_indices_from_world_extrema(
            extrema, self.shape
        )

        # NOTE : The data needs matrix indexing, hence y is on the first axis.
        return self.dm.data[miny : maxy + 1, minx : maxx + 1]

    def std_inside_extrema(self, extrema: SkyCoord) -> ArrayLike:
        """Find the data values inside the extrema.

        Parameters
        ----------
        extrema : List[SkyCoord]
            List of SkyCoord objects, representing the world location of the
            reconstruction grid edges.

        Returns
        -------
        data : ArrayLike
            Data values inside the extrema.
        """
        # NOTE : Carefull replacing bounding_box_indices_from_world_extrema since this
        # is coupled inside self.load_cutout_data_mask_std_by_world_corners.

        minx, maxx, miny, maxy = self.wcs.bounding_box_indices_from_world_extrema(
            extrema, self.shape
        )
        # NOTE : The data needs matrix indexing, hence y is on the first axis.
        return self.dm.err[miny : maxy + 1, minx : maxx + 1]

    def nan_inside_extrema(self, extrema: SkyCoord) -> ArrayLike:
        """
        Get a nan-mask of the data inside the extrema.

        Parameters
        ----------
        extrema : List[SkyCoord]
            List of SkyCoord objects, representing the world location of the
            reconstruction grid edges.

        Returns
        -------
        nan-mask : ArrayLike
            Mask corresponding to the nan values inside the extrema.
        """
        # NOTE : Carefull replacing bounding_box_indices_from_world_extrema since this
        # is coupled inside self.load_cutout_data_mask_std_by_world_corners.

        minx, maxx, miny, maxy = self.wcs.bounding_box_indices_from_world_extrema(
            extrema, self.shape
        )
        # NOTE : The data needs matrix indexing, hence y is on the first axis.
        return (~np.isnan(self.dm.data[miny : maxy + 1, minx : maxx + 1])) * (
            ~np.isnan(self.dm.err[miny : maxy + 1, minx : maxx + 1])
        )

    @property
    def half_power_wavelength(self):
        pivot, bw, er, blue, red = JWST_FILTERS[self.filter]
        return ColorRange(Color(blue * units.micrometer), Color(red * units.micrometer))

    @property
    def pivot_wavelength(self):
        pivot, *_ = JWST_FILTERS[self.filter]
        return Color(pivot * units.micrometer)

    @property
    def transmission(self):
        """
        Effective response is the mean transmission value over the
        wavelength range.

        see:
        https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-instrumentation/miri-filters-and-dispersers#gsc.tab=0
        """
        pivot, bw, effective_response, blue, red = JWST_FILTERS[self.filter]
        return effective_response

    def bounding_data_mask_std_by_world_corners(
        self,
        reconstruction_grid: Grid,
        world_corners: list[SkyCoord],
        additional_masks_corners: list[CornerMaskConfig],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a subpart of the data, mask, and std inside a bounding box surrounding
        the `world_corners`.

        Parameters
        ----------
        world_corners: list[SkyCoord]
            The world corners of the cutout. I.e. the absolute positions in the world.
        additional_masks_corners: list[CornerMaskConfig]
            Holds the egde points of additional masks for the data.

        Returns
        -------
        data: np.ndarray[float],
        mask: np.ndarray[bool],
        std: np.ndarray[float]

        Notes
        -----
        The mask is true where the data will be taken, i.e. supplied to the likelihood.
        """
        # bounding_world_corners: SkyCoord
        #     The world coordinates of the pixel edges of the bounding box (i.e. the data)

        # NOTE : Carefull replacing bounding_box_indices_from_world_extrema since this
        # is coupled inside self.data_inside_extrema, self.nan_inside_extrema, and
        # self.std_inside_extrema.

        # xmin, xmax, ymin, ymax = np.array(
        #     self.wcs.bounding_box_indices_from_world_extrema(world_corners)
        # )
        # points = np.array(((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)))
        # cutout_world_corners = self.wcs.pixel_to_world(*np.array(points).T)

        data = self.data_inside_extrema(world_corners)
        mask = get_mask_from_index_centers_within_rgrid(
            world_corners, self.wcs, reconstruction_grid.spatial
        )
        mask *= self.nan_inside_extrema(world_corners)

        extra_masks = [
            get_mask_from_mask_corners(data.shape, self.wcs, world_corners, mc.corners)
            for mc in additional_masks_corners
        ]
        mask *= ~np.sum(extra_masks, axis=0, dtype=bool)
        std = self.std_inside_extrema(world_corners)

        return data, mask, std

    def get_boresight_world_coords(self):
        """
        Returns the most accurate world coordinate system point of the boresight (v1) from a JWST datamodel.
        This function considers possible corrections such as velocity aberration corrections if available.
        """
        # Check available WCS frames
        available_frames = self.wcs.available_frames

        # Use v2v3vacorr if available for the most accurate V2/V3 coordinates, else fallback to v2v3
        if "v2v3vacorr" in available_frames:
            v2v3_frame = "v2v3vacorr"
        elif "v2v3" in available_frames:
            v2v3_frame = "v2v3"
        else:
            raise ValueError("No V2V3 coordinate frame available in the model")

        # The boresight in V2V3 coordinates is at (0, 0) by definition
        v2, v3 = 0.0, 0.0

        # Transform from V2V3 to world coordinates (RA, Dec)
        ra, dec = self.wcs.transform(v2v3_frame, "world", v2, v3)
        return SkyCoord(ra, dec, unit="deg")
