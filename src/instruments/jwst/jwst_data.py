# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %

from ...color import Color, ColorRange
from ...grid import Grid
from ...wcs.wcs_jwst_data import WcsJwstData
from ...wcs.wcs_subsample_centers import subsample_grid_centers_in_index_grid
from .masking import (
    get_mask_from_index_centers_within_rgrid,
    get_mask_from_mask_corners,
)
from .parse.masking.data_mask import CornerMaskConfig

from astropy import units
from astropy.coordinates import SkyCoord

try:
    from jwst import datamodels
except ImportError:
    print("jwst not installed. Some JWST functions will not work.")
    pass

import numpy as np
from numpy.typing import ArrayLike


nircam_filters = dict(
    # https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-instrumentation/miri-filters-and-dispersers#gsc.tab=0
    # Pivot λ (μm), BW Δλ (μm), Effective response, Blue λ- (µm), Red λ+ (µm)
    F070W=(0.704, 0.128, 0.237, 0.624, 0.781),
    F090W=(0.901, 0.194, 0.318, 0.795, 1.005),
    F115W=(1.154, 0.225, 0.333, 1.013, 1.282),
    F140M=(1.404, 0.142, 0.434, 1.331, 1.479),
    F150W=(1.501, 0.318, 0.476, 1.331, 1.668),
    F162M=(1.626, 0.168, 0.469, 1.542, 1.713),
    F164N=(1.644, 0.020, 0.385, 1.635, 1.653),
    F150W2=(1.671, 1.227, 0.489, 1.007, 2.38),
    F182M=(1.845, 0.238, 0.505, 1.722, 1.968),
    F187N=(1.874, 0.024, 0.434, 1.863, 1.885),
    F200W=(1.990, 0.461, 0.525, 1.755, 2.227),
    F210M=(2.093, 0.205, 0.522, 1.992, 2.201),
    F212N=(2.120, 0.027, 0.420, 2.109, 2.134),
    F250M=(2.503, 0.181, 0.370, 2.412, 2.595),
    F277W=(2.786, 0.672, 0.412, 2.423, 3.132),
    F300M=(2.996, 0.318, 0.432, 2.831, 3.157),
    F322W3=(3.247, 1.339, 0.499, 2.432, 4.013),
    F323N=(3.237, 0.038, 0.290, 3.217, 3.255),
    F335M=(3.365, 0.347, 0.480, 3.177, 3.537),
    F356W=(3.563, 0.787, 0.530, 3.135, 3.981),
    F360M=(3.621, 0.372, 0.515, 3.426, 3.814),
    F405N=(4.055, 0.046, 0.418, 4.030, 4.076),
    F410M=(4.092, 0.436, 0.499, 3.866, 4.302),
    F430M=(4.280, 0.228, 0.526, 4.167, 4.398),
    F444W=(4.421, 1.024, 0.533, 3.881, 4.982),
    F460M=(4.624, 0.228, 0.460, 4.515, 4.747),
    F466N=(4.654, 0.054, 0.320, 4.629, 4.681),
    F470N=(4.707, 0.051, 0.316, 4.683, 4.733),
    F480M=(4.834, 0.303, 0.447, 4.662, 4.973),
)

miri_filters = dict(
    # https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-instrumentation/miri-filters-and-dispersers#gsc.tab=0
    # Pivot λ (μm), BW Δλ (μm), Effective response, Blue λ- (µm), Red λ+ (µm)
    F560W=(5.635, 1.00, 0.245, 5.054, 6.171),
    F770W=(7.639, 1.95, 0.355, 6.581, 8.687),
    F1000W=(9.953, 1.80, 0.466, 9.023, 10.891),
    F1130W=(11.309, 0.73, 0.412, 10.953, 11.667),
    F1280W=(12.810, 2.47, 0.384, 11.588, 14.115),
    F1500W=(15.064, 2.92, 0.442, 13.527, 16.640),
    F1800W=(17.984, 2.95, 0.447, 16.519, 19.502),
    F2100W=(20.795, 4.58, 0.352, 18.477, 23.159),
    F2550W=(25.365, 3.67, 0.269, 23.301, 26.733),
    F2550WR=(25.365, 3.67, 0.269, 23.301, 26.733),
)

JWST_FILTERS = nircam_filters | miri_filters


# https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument#gsc.tab=0
# https://iopscience.iop.org/article/10.1086/682254
# https://jwst-docs.stsci.edu/jwst-near-infrared-camera#gsc.tab=0
MIRI_PIXEL_DISTANCE = 0.11 * units.arcsec
NIRCAM_PIXEL_DISTANCE_01 = 0.031 * units.arcsec  # 0.6–2.3 µm wavelength range
NIRCAM_PIXEL_DISTANCE_02 = 0.063 * units.arcsec  # 2.4–5.0 µm wavelength range


def _get_dvol(filter: str):
    if filter in miri_filters:
        return (MIRI_PIXEL_DISTANCE.to(units.deg)) ** 2

    elif filter in nircam_filters:
        pivot = Color(nircam_filters[filter][0] * units.micrometer)

        if pivot in ColorRange(
            Color(0.6 * units.micrometer), Color(2.3 * units.micrometer)
        ):
            # 0.6–2.3 µm wavelength range
            return NIRCAM_PIXEL_DISTANCE_01.to(units.deg) ** 2

        else:
            # 2.4–5.0 µm wavelength range
            return NIRCAM_PIXEL_DISTANCE_02.to(units.deg) ** 2

    else:
        raise NotImplementedError(
            f"filter has to be in the supported filters{JWST_FILTERS.keys()}"
        )


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
        self.dvol = _get_dvol(self.filter)

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
