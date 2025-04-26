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
from ...wcs.wcs_astropy import WcsAstropy
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
        reconstruction_grid_wcs: WcsAstropy,
        world_corners: list[SkyCoord],
        additional_masks_corners: list[CornerMaskConfig] | None = None,
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
            world_corners, self.wcs, reconstruction_grid_wcs
        )
        mask *= self.nan_inside_extrema(world_corners)
        std = self.std_inside_extrema(world_corners)

        if additional_masks_corners is not None:
            extra_masks = [
                get_mask_from_mask_corners(
                    data.shape, self.wcs, world_corners, mc.corners
                )
                for mc in additional_masks_corners
            ]
            mask *= ~np.sum(extra_masks, axis=0, dtype=bool)

        return data, mask, std

    def get_boresight_world_coords(self):
        """Returns the world coordinate of the boresight (v1) from a JWST datamodel."""
        return SkyCoord(
            self.dm.meta.pointing.ra_v1, self.dm.meta.pointing.dec_v1, unit="deg"
        )
