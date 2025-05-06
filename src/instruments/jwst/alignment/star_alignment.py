from dataclasses import dataclass, field

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

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

    def subpixel_position_in_world_coordinates(self, world_coordinates: SkyCoord):
        return self.skycoord_to_subpixel(self.position, world_coordinates)

    @staticmethod
    def skycoord_to_subpixel(position: SkyCoord, world_coordinates: SkyCoord):
        """Calculate the (sub)pixel position of `position` in the `world_coordinates` grid.

        Parameters
        ----------
        position : SkyCoord                         # the star
        world_coordinates   : SkyCoord (ny, nx)     # pixel-centre grid

        Returns
        -------
        y_pix, x_pix : float
            0-based pixel coordinates whose fractional part gives the sub-pixel
            offset (0.0 = lower edge, 0.5 = centre, 1.0 = upper edge).
        """
        # --- 1. which pixel centre is closest? -------------------------------
        sep = position.separation(world_coordinates)  # great-circle distance [[2]]
        i0, j0 = np.unravel_index(np.argmin(sep), world_coordinates.shape)
        centre = world_coordinates[i0, j0]

        # --- 2. build a tiny tangent plane around that pixel -----------------
        off_frame = centre.skyoffset_frame()  # gnomonic projection
        star_off = position.transform_to(off_frame)

        # --- 3. estimate one-pixel steps along x and y -----------------------
        # protect against edges of the array
        j1 = j0 + 1 if j0 < world_coordinates.shape[1] - 1 else j0 - 1
        i1 = i0 + 1 if i0 < world_coordinates.shape[0] - 1 else i0 - 1

        dx = world_coordinates[i0, j1].transform_to(off_frame).lon.to(u.arcsec).value
        dy = world_coordinates[i1, j0].transform_to(off_frame).lat.to(u.arcsec).value
        scale_x = abs(dx)  # arcsec per pixel along x
        scale_y = abs(dy)  # arcsec per pixel along y

        # --- 4. fractional offset inside the host pixel ----------------------
        frac_x = star_off.lon.to(u.arcsec).value / scale_x  # −0.5 … +0.5
        frac_y = star_off.lat.to(u.arcsec).value / scale_y

        # --- 5. full floating-point pixel coordinate -------------------------
        x_pix = j0 + 0.5 + frac_x  # 0-based; centre sits at *.5
        y_pix = i0 + 0.5 + frac_y

        return y_pix, x_pix


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
