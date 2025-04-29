# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %
from dataclasses import dataclass

import numpy as np
from astropy import units
from astropy.coordinates import SkyCoord
from numpy.typing import ArrayLike

from ....color import Color, ColorRange
from ....wcs import subsample_pixel_centers, WcsAstropy, WcsJwstData

from .jwst_information import get_dvol, JWST_FILTERS, get_pixel_distance
from ..masking import (
    get_mask_from_index_centers_within_rgrid,
    get_mask_from_mask_corners,
)
from ..parse.masking.data_mask import CornerMaskConfig

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
    pixel_distance: units.Quantity


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
            pixel_distance=get_pixel_distance(self.filter),
        )

    def _handle_extrema(
        self,
        extrema: SkyCoord | tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        if isinstance(extrema, SkyCoord):
            return self.wcs.bounding_box_indices_from_world_extrema(extrema, self.shape)
        elif isinstance(extrema, tuple) or isinstance(extrema, np.ndarray):
            assert len(extrema) == 4
            return extrema

        raise ValueError(
            "The extrema have to be given as `astropy.coordinates.SkyCoord` or "
            "tuple | ndarray holding `minx_maxx_miny_maxy`."
        )

    def data_inside_extrema(
        self, extrema: SkyCoord | tuple[int, int, int, int]
    ) -> ArrayLike:
        """
        Find the data values inside the extrema.

        Parameters
        ----------
        extrema : SkyCoord | tuple[int, int, int, int]
            If SkyCoord, the values represent the world location of the grid corners.
            If tuple[int], the values are assumed to be (minx, maxx, miny, maxy) which
            are the borders of the data.

        Returns
        -------
        data : ArrayLike
            Data values inside the extrema.
        """
        minx, maxx, miny, maxy = self._handle_extrema(extrema)

        # NOTE : The data needs matrix indexing, hence y is on the first axis.
        return self.dm.data[miny : maxy + 1, minx : maxx + 1]

    def std_inside_extrema(
        self, extrema: SkyCoord | tuple[int, int, int, int]
    ) -> ArrayLike:
        """Find the data values inside the extrema.

        Parameters
        ----------
        extrema : SkyCoord | tuple[int, int, int, int]
            If SkyCoord, the values represent the world location of the grid corners.
            If tuple[int], the values are assumed to be (minx, maxx, miny, maxy) which
            are the borders of the data.

        Returns
        -------
        data : ArrayLike
            Data values inside the extrema.
        """

        minx, maxx, miny, maxy = self._handle_extrema(extrema)
        # NOTE : The data needs matrix indexing, hence y is on the first axis.
        return self.dm.err[miny : maxy + 1, minx : maxx + 1]

    def nan_inside_extrema(
        self, extrema: SkyCoord | tuple[int, int, int, int]
    ) -> ArrayLike:
        """
        Get a nan-mask of the data inside the extrema.

        Parameters
        ----------
        extrema : SkyCoord | tuple[int, int, int, int]
            If SkyCoord, the values represent the world location of the grid corners.
            If tuple[int], the values are assumed to be (minx, maxx, miny, maxy) which
            are the borders of the data.

        Returns
        -------
        nan-mask : ArrayLike
            Mask corresponding to the nan values inside the extrema.
        """

        minx, maxx, miny, maxy = self._handle_extrema(extrema)
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

    def bounding_data_mask_std_subpixel_by_world_corners(
        self,
        reconstruction_grid_wcs: WcsAstropy,
        world_corners: list[SkyCoord],
        additional_masks_corners: list[CornerMaskConfig] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a subpart of the data, mask, and std inside a bounding box surrounding
        the `world_corners`.

        Parameters
        ----------
        reconstruction_grid_wcs: WcsAstropy,
            The wcs of the reconstruction/index grid.
        world_corners: list[SkyCoord]
            The world corners of the cutout. I.e. the absolute positions in the world.
        additional_masks_corners: list[CornerMaskConfig]
            Holds the egde points of additional masks for the data.

        Returns
        -------
        data: np.ndarray[float],
        mask: np.ndarray[bool],
        std: np.ndarray[float]
        subsampled_pixel_centers: SkyCoord
            The world coordinates of the subsampled data pixels.

        Notes
        -----
        The mask is true where the data will be taken, i.e. supplied to the likelihood.
        """

        xmin_xmax_ymin_ymax = np.array(
            self.wcs.bounding_box_indices_from_world_extrema(world_corners)
        )
        return self.bounding_data_mask_std_subpixel_by_bounding_indices(
            reconstruction_grid_wcs=reconstruction_grid_wcs,
            bounding_box_xmin_xmax_ymin_ymax=xmin_xmax_ymin_ymax,
            additional_masks_corners=additional_masks_corners,
        )

    def bounding_data_mask_std_subpixel_by_bounding_indices(
        self,
        reconstruction_grid_wcs: WcsAstropy,
        bounding_box_xmin_xmax_ymin_ymax: tuple[int] | np.ndarray,
        additional_masks_corners: list[CornerMaskConfig] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a subpart of the data, mask, and std inside a bounding box surrounding
        the `world_corners`.

        Parameters
        ----------
        reconstruction_grid_wcs: WcsAstropy,
            The wcs of the reconstruction/index grid.
        bounding_box_xmin_xmax_ymin_ymax: tuple[int] | np.ndarray
            The pixel postions edges of the bounding box, inside the data (corresponding
            from the data_wcs).
            Previously cacuclated by,
            data_wcs.bounding_box_indices_from_world_extrema(world_corners).
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

        data_subsampled_centers = subsample_pixel_centers(
            bounding_box_xmin_xmax_ymin_ymax=bounding_box_xmin_xmax_ymin_ymax,
            to_be_subsampled_grid_wcs=self.wcs,
            subsample=self.meta.subsample,
        )

        data = self.data_inside_extrema(bounding_box_xmin_xmax_ymin_ymax)
        mask = get_mask_from_index_centers_within_rgrid(
            bounding_box_xmin_xmax_ymin_ymax, self.wcs, reconstruction_grid_wcs
        )
        mask *= self.nan_inside_extrema(bounding_box_xmin_xmax_ymin_ymax)
        std = self.std_inside_extrema(bounding_box_xmin_xmax_ymin_ymax)

        if additional_masks_corners is not None:
            extra_masks = [
                get_mask_from_mask_corners(
                    data.shape, self.wcs, bounding_box_xmin_xmax_ymin_ymax, mc.corners
                )
                for mc in additional_masks_corners
            ]
            mask *= ~np.sum(extra_masks, axis=0, dtype=bool)

        return data, mask, std, data_subsampled_centers

    def get_boresight_world_coords(self):
        """Returns the world coordinate of the boresight (v1) from a JWST datamodel."""
        return SkyCoord(
            self.dm.meta.pointing.ra_v1, self.dm.meta.pointing.dec_v1, unit="deg"
        )
