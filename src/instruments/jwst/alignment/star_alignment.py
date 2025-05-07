from dataclasses import dataclass, field

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from ....wcs.wcs_jwst_data import WcsJwstData
from ....wcs.wcs_astropy import WcsAstropy
from ...gaia.star_finder import join_tables
from ..data.jwst_data import JwstData
from ..parse.alignment.star_alignment import FilterAlignmentMeta
from ..parse.parametric_model.parametric_prior import (
    prior_config_factory,
)
from ..parse.rotation_and_shift.coordinates_correction import (
    ROTATION_KEY,
    ROTATION_UNIT_KEY,
    SHIFT_KEY,
    SHIFT_UNIT_KEY,
    CoordinatesCorrectionPriorConfig,
)

DEFAULT_KEY = "default"


@dataclass
class Star:
    id: int
    position: SkyCoord

    def __getitem__(self, index: int):
        return Star(self.id[index], self.position[index])

    def bounding_indices(
        self, jwst_data: JwstData, shape: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        pixel_position = jwst_data.wcs.world_to_pixel(self.position)
        return self._get_bounding_indices(pixel_position, shape, jwst_data.shape)

    @staticmethod
    def _get_bounding_indices(
        pixel_position: tuple[float, float],
        shape: tuple[int, int],
        jwst_data_shape: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        for sh in shape:
            assert sh % 2 != 0, (
                "Provide uneven pixel shapes for the star alignment cutouts."
            )

        pp = [int(t) for t in np.floor(pixel_position)]
        half = [(sh - 1) // 2 for sh in shape]

        minx = max(pp[0] - half[0], 0)
        miny = max(pp[1] - half[1], 0)
        maxx = min(pp[0] + half[0], jwst_data_shape[0])
        maxy = min(pp[1] + half[1], jwst_data_shape[0])

        return (minx, maxx, miny, maxy)

    def pixel_position_in_subsampled_data(
        self,
        data_wcs: WcsJwstData | WcsAstropy,
        min_row: int = 0,
        min_column: int = 0,
        subsample_factor: int = 1,
    ):
        """Get the pixel position of the star position in the (subsampled) data grid.

        Parameters
        ----------
        data_wcs : WcsJwstData | WcsAstropy
            The wcs of the data.
        min_row: int, optional
            The row index of the first pixel for a subpart of the data.
        min_column: int, optional
            The column index of the first pixel for a subpart of the data.
        subsample_factor: int
            The factor by which the pixels will be subsampled.

        Returns
        -------
        x_pix, y_pix : tuple[float]
            The pixel position of the star in the subsampled data.
            (0.0, 0.0)   = pixel center of the upper-left corner of the grid.
            (-0.5, -0.5) = upper-left corner of the grid.
        """
        return self.skycoord_to_subpixel(
            self.position,
            data_wcs,
            min_row=min_row,
            min_column=min_column,
            subsample_factor=subsample_factor,
        )

    @staticmethod
    def skycoord_to_subpixel(
        position: SkyCoord,
        data_wcs: WcsJwstData | WcsAstropy,
        min_row: int = 0,
        min_column: int = 0,
        subsample_factor: int = 1,
    ):
        """Calculate the (sub)pixel position of `position` in the `data_wcs`.
        If `subsample_factor` is provided, this will

        Parameters
        ----------
        position : SkyCoord
            The position of the star.
        data_wcs : WcsJwstData | WcsAstropy
            The wcs of the data.
        min_row: int, optional
            The row index of the first pixel for a subpart of the data.
        min_column: int, optional
            The column index of the first pixel for a subpart of the data.
        subsample_factor : int, optional
            Factor by which each detector pixel is divided. `1` (default) means native
            pixels, `N` means every pixel is replaced with an `NxN` finer grid.

        Returns
        -------
        x_pix, y_pix : tuple[float]
            The pixel position of the star in the subsampled data.
            (0.0, 0.0)   = pixel center of the upper-left corner of the grid.
            (-0.5, -0.5) = upper-left corner of the grid.
        """

        x_detector, y_detector = data_wcs.world_to_pixel(position)
        x_detector -= min_column
        y_detector -= min_row

        if subsample_factor != 1:
            x_detector *= subsample_factor
            y_detector *= subsample_factor

        return float(x_detector), float(y_detector)


@dataclass
class FilterAlignment:
    filter_name: str
    alignment_meta: FilterAlignmentMeta
    correction_prior: CoordinatesCorrectionPriorConfig | None = None
    star_tables: list[Table] = field(default_factory=list)
    boresight: list[SkyCoord] = field(default_factory=list)

    def get_stars(self, observation_id: int | None = None) -> list[Star]:
        if observation_id is not None:
            table = self.star_tables[observation_id]
        else:
            table = join_tables(self.star_tables)

        source_id = table["SOURCE_ID"]
        positions = SkyCoord(ra=table["ra"], dec=table["dec"], unit="deg")

        return [
            Star(id, position)
            for id, position in zip(source_id, positions)
            if id not in self.alignment_meta.exclude_source_id
        ]

    def load_correction_prior(self, raw: dict, number_of_observations: int):
        if self.filter_name in raw:
            config = raw[self.filter_name]
        else:
            config = raw[DEFAULT_KEY]

        self.correction_prior = CoordinatesCorrectionPriorConfig(
            shift=prior_config_factory(
                config[SHIFT_KEY], shape=(number_of_observations, 2)
            ),
            rotation=prior_config_factory(
                config[ROTATION_KEY], shape=(number_of_observations, 1)
            ),
            shift_unit=getattr(u, raw[SHIFT_UNIT_KEY]),
            rotation_unit=getattr(u, raw[ROTATION_UNIT_KEY]),
        )
