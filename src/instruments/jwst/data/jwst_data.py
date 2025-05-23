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
from ..parse.masking.data_mask import ExtraMaskFromCorners

try:
    from jwst import datamodels
except ImportError:
    print("jwst not installed. Some JWST functions will not work.")
    pass


@dataclass
class DataMetaInformation:
    unit: units.Unit
    dvol: units.Quantity
    pixel_scale: units.Quantity
    color: Color


class JwstData:
    """Class to contain JWST data metadata."""

    def __init__(self, filepath: str):
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
            unit=units.Unit(self.dm.meta.bunit_data),
            dvol=get_dvol(self.filter),
            pixel_scale=get_pixel_distance(self.filter),
            color=self.pivot_wavelength,
        )

    def data_from_bounding_indices(
        self, min_row: int, max_row: int, min_column: int, max_column: int
    ) -> ArrayLike:
        """
        Find the data values inside the extrema.

        Parameters
        ----------
        min_row: int
        max_row: int
        min_column: int
        max_column: int

        Returns
        -------
        data : ArrayLike
            Data values inside the extrema.
        """
        return self.dm.data[min_row:max_row, min_column:max_column]

    def std_from_bounding_indices(
        self, min_row: int, max_row: int, min_column: int, max_column: int
    ) -> ArrayLike:
        """Find the data values inside the extrema.

        Parameters
        ----------
        min_row: int
        max_row: int
        min_column: int
        max_column: int

        Returns
        -------
        std : ArrayLike
            Data values inside the extrema.
        """
        return self.dm.err[min_row:max_row, min_column:max_column]

    def nan_from_bounding_indices(
        self, min_row: int, max_row: int, min_column: int, max_column: int
    ) -> ArrayLike:
        """
        Get a nan-mask of the data inside the extrema.

        Parameters
        ----------
        min_row: int
        max_row: int
        min_column: int
        max_column: int

        Returns
        -------
        nan-mask : ArrayLike
            Mask corresponding to the nan values inside the extrema.
        """
        data = self.data_from_bounding_indices(min_row, max_row, min_column, max_column)
        std = self.std_from_bounding_indices(min_row, max_row, min_column, max_column)
        return (~np.isnan(data)) * (~np.isnan(std))

    @property
    def half_power_wavelength(self):
        pivot, bw, er, blue, red = JWST_FILTERS[self.filter]
        return ColorRange(Color(blue * units.micrometer), Color(red * units.micrometer))

    @property
    def pivot_wavelength(self) -> Color:
        pivot, *_ = JWST_FILTERS[self.filter]
        return Color(pivot * units.micrometer)

    def bounding_data_mask_std_by_bounding_indices(
        self,
        row_minmax_column_minmax: tuple[int] | np.ndarray,
        reconstruction_grid_wcs: WcsAstropy | None = None,
        additional_masks_corners: list[ExtraMaskFromCorners] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Data, mask, and std cutout corresponding to `row_minmax_column_minmax`.

        Parameters
        ----------
        row_minmax_column_minmax: tuple[int] | np.ndarray
            The slice indices, corresponding to a bounding box for the data.
        reconstruction_grid_wcs: WcsAstropy | None, optional
            The wcs of the reconstruction/index grid. Used for getting the mask, where
            the reconstruction grid has values. If None no value will be masked from the
            grid.
        additional_masks_corners: list[CornerMaskConfig] | None, optional
            Holds the egde points of additional masks for the data.

        Returns
        -------
        data: np.ndarray[float]
        mask: np.ndarray[bool]
        std: np.ndarray[float]

        Notes
        -----
        The mask is true where the data will be taken, i.e. supplied to the likelihood.
        """

        data = self.data_from_bounding_indices(*row_minmax_column_minmax)
        mask = get_mask_from_index_centers_within_rgrid(
            row_minmax_column_minmax, self.wcs, reconstruction_grid_wcs
        )
        mask *= self.nan_from_bounding_indices(*row_minmax_column_minmax)
        std = self.std_from_bounding_indices(*row_minmax_column_minmax)

        if additional_masks_corners is not None:
            extra_masks = [
                get_mask_from_mask_corners(
                    data.shape, self.wcs, row_minmax_column_minmax, mc.corners
                )
                for mc in additional_masks_corners
            ]
            mask *= ~np.sum(extra_masks, axis=0, dtype=bool)

        return data, mask, std

    def data_subpixel_centers(
        self,
        row_minmax_column_minmax: tuple[int] | np.ndarray,
        subsample: int = 1,
    ):
        """Subsampled data pixel centers according to `row_minmax_column_minmax`.

        Parameters
        ----------
        row_minmax_column_minmax: tuple[int] | np.ndarray
            The slice indices, corresponding to a bounding box for the data.
        subsample: int, optional
            The multiplicity of the subsampling along each axis. How many sub-pixels
            will a single pixel in the to_be_subsampled_grid have along each axis. `1`
            will correspond to no subsampling, i.e. the world coordinates of the
            data pixel centers.
        """
        return subsample_pixel_centers(
            bounding_indices=row_minmax_column_minmax,
            to_be_subsampled_grid_wcs=self.wcs,
            subsample=subsample,
        )

    def get_boresight_world_coords(self):
        """Returns the world coordinate of the boresight (v1) from a JWST datamodel."""
        return SkyCoord(
            self.dm.meta.pointing.ra_v1, self.dm.meta.pointing.dec_v1, unit="deg"
        )

    def get_reference_pixel_world_coords(self):
        """Returns the world coordinate of the boresight (v1) from a JWST datamodel."""
        ra, dec = self.dm.meta.wcsinfo.ra_ref, self.dm.meta.wcsinfo.dec_ref
        return SkyCoord(ra, dec, unit="deg")
