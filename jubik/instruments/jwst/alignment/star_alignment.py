from dataclasses import dataclass
from itertools import combinations

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astropy.time import Time


from ..wcs_jwst_data import WcsJwstData
from ....wcs.wcs_astropy import WcsAstropy
from ..data.jwst_data import JwstData


@dataclass
class Star:
    """A Gaia star: its astrometric solution at the Gaia epoch.

    Parameters
    ----------
    id: int
        The Gaia source id.
    astrometry: SkyCoord
        The Gaia position and proper motion, at the Gaia epoch `obstime`.
    astrometric_covariance: np.ndarray | None
        The Gaia covariance of (RA cos(dec), Dec, pmRA cos(dec), pmDec) in mas and
        mas/yr, shape (4, 4). NaN where a Gaia error is masked, None if the table
        has no error columns.
    """

    id: int
    astrometry: SkyCoord
    astrometric_covariance: np.ndarray | None = None

    def position_at(self, time: Time | str) -> SkyCoord:
        """The star position at `time` (UTC)."""
        return self.astrometry.apply_space_motion(Time(time, scale="utc"))

    def epoch_jacobian(self, time: Time | str) -> np.ndarray:
        """Change of the (RA cos(dec), Dec) offset from `position_at(time)` per
        change of the astrometry, in mas per (mas, mas/yr), shape (2, 4).

        Central differences of `position_at`, so it follows the motion on the
        sphere; the linear `position + dt * proper motion` breaks down within
        arcseconds of the pole, where the local frame turns between epochs."""
        a = self.astrometry
        steps = np.concatenate([np.eye(4), -np.eye(4)])  # 1 mas, 1 mas/yr
        moved = SkyCoord(
            ra=a.ra + steps[:, 0] * u.mas / np.cos(a.dec),
            dec=a.dec + steps[:, 1] * u.mas,
            pm_ra_cosdec=a.pm_ra_cosdec + steps[:, 2] * u.mas / u.yr,
            pm_dec=a.pm_dec + steps[:, 3] * u.mas / u.yr,
            obstime=a.obstime,
        ).apply_space_motion(Time(time, scale="utc"))
        offsets = np.array(
            [
                o.to_value(u.mas)
                for o in self.position_at(time).spherical_offsets_to(moved)
            ]
        )
        return (offsets[:, :4] - offsets[:, 4:]) / 2

    def bounding_indices(
        self, jwst_data: JwstData, shape: tuple[int, int], time: Time | str
    ) -> tuple[int, int, int, int]:
        """Get the bounding pixel slice for the star position.

        Parameters
        ----------
        jwst_data: JwstData
            The jwst observation containing the star.
        shape: tuple
            The shape of the cutout.
        time: Time | str
            The epoch of the observation.

        Returns
        -------
        (min_row, max_row, min_column, max_column) of the cutout.
        """
        pixel_position = jwst_data.wcs.world_to_pixel(self.position_at(time))
        return self._get_bounding_indices(pixel_position, shape, jwst_data.shape)

    @staticmethod
    def _get_bounding_indices(
        pixel_position: tuple[float, float],
        shape: tuple[int, int],
        jwst_data_shape: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        """
        Return (y_min, y_max, x_min, x_max) so that
        frame[y_min:y_max, x_min:x_max] has exactly `shape`
        and is fully contained in the image.
        """
        # ---------------------------------------------------------------------
        # 1. Sanity checks
        # ---------------------------------------------------------------------
        if len(shape) != 2:
            raise ValueError("shape must be a length-2 tuple")
        if any(s % 2 == 0 for s in shape):
            raise ValueError("Provide *odd* dimensions so the star stays centred.")

        ny_frame, nx_frame = jwst_data_shape
        ny_win, nx_win = shape

        # ---------------------------------------------------------------------
        # 2. Integer centre pixel (row, col)
        # ---------------------------------------------------------------------
        cx, cy = (int(np.floor(pixel_position[0])), int(np.floor(pixel_position[1])))

        # pixels to each side of the centre
        hy, hx = (ny_win - 1) // 2, (nx_win - 1) // 2

        # initial bounds  (upper bound is *exclusive*)
        y_min, y_max = cy - hy, cy + hy + 1
        x_min, x_max = cx - hx, cx + hx + 1

        # ---------------------------------------------------------------------
        # 3. Shift window back inside the frame (keeps full size)
        # ---------------------------------------------------------------------
        if y_min < 0:  # too far up      → push down
            y_max += -y_min
            y_min = 0
        if y_max > ny_frame:  # too far down    → pull up
            y_min -= y_max - ny_frame
            y_max = ny_frame

        if x_min < 0:  # too far left    → push right
            x_max += -x_min
            x_min = 0
        if x_max > nx_frame:  # too far right   → pull left
            x_min -= x_max - nx_frame
            x_max = nx_frame

        # ---------------------------------------------------------------------
        # 4. Final guarantees
        # ---------------------------------------------------------------------
        assert 0 <= y_min < y_max <= ny_frame
        assert 0 <= x_min < x_max <= nx_frame
        assert (y_max - y_min, x_max - x_min) == (ny_win, nx_win)

        return y_min, y_max, x_min, x_max

    def pixel_position_in_subsampled_data(
        self,
        data_wcs: WcsJwstData | WcsAstropy,
        time: Time | str,
        min_row: int = 0,
        min_column: int = 0,
        subsample_factor: int = 1,
    ) -> tuple[float, float]:
        """Get the pixel position of the star position in the (subsampled) data grid.

        Parameters
        ----------
        data_wcs : WcsJwstData | WcsAstropy
            The wcs of the data.
        time: Time | str
            The epoch of the observation.
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
            self.position_at(time),
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

        # Subpixel m of native pixel n is centered at n + (m + 0.5) / s - 0.5
        # (`subsample_pixel_centers`), so fine index k = n * s + m inverts as below.
        x_detector = (x_detector + 0.5) * subsample_factor - 0.5
        y_detector = (y_detector + 0.5) * subsample_factor - 0.5

        return float(x_detector), float(y_detector)


# Gaia's astrometric order; the `*_corr` columns name pairs in this order.
_GAIA_ASTROMETRY = ("ra", "dec", "pmra", "pmdec")
_GAIA_UNITS = (u.mas, u.mas, u.mas / u.yr, u.mas / u.yr)


def _nan_filled(column) -> u.Quantity:
    """The column with masked entries as NaN, not the 0 that `u.Quantity` reads."""
    return u.Quantity(np.ma.filled(column, np.nan))


def _astrometric_covariance(table: Table) -> np.ndarray | list[None]:
    """Gaia covariance of (RA cos(dec), Dec, pmRA cos(dec), pmDec) in mas and
    mas/yr, shape (n, 4, 4), from the Gaia errors and correlations; NaN where an
    entry is masked. None per star if the table lacks the error columns."""
    pairs = list(combinations(enumerate(_GAIA_ASTROMETRY), 2))
    names = [f"{n}_error" for n in _GAIA_ASTROMETRY]
    names += [f"{a}_{b}_corr" for (_, a), (_, b) in pairs]
    if not set(names) <= set(table.colnames):
        return [None] * len(table)

    def column(name, unit=u.one):
        return _nan_filled(table[name]).to_value(unit)

    std = np.stack(
        [column(f"{n}_error", unit) for n, unit in zip(_GAIA_ASTROMETRY, _GAIA_UNITS)],
        axis=-1,
    )
    correlation = np.tile(np.eye(4), (len(table), 1, 1))
    for (i, a), (j, b) in pairs:
        correlation[:, i, j] = correlation[:, j, i] = column(f"{a}_{b}_corr")
    return std[:, :, None] * correlation * std[:, None, :]


class StarTables:
    def __init__(
        self,
        data: list[Table] | None = None,
        observation_times: list[Time | str] | None = None,
    ):
        self.data = data if data else []
        self.observation_times = observation_times if observation_times else []

    def get_stars(self, observation_id: int | None = None) -> list[Star]:
        if observation_id is not None:
            table = self.data[observation_id]
        else:
            table = self.join_tables(self.data)

        if len(table) == 0:
            return []

        source_id = table["SOURCE_ID"]
        # No distance on purpose. Gaia DR3 contains negative and missing
        # parallaxes; `Distance(parallax=...)` turns those into NaN distances,
        # which propagate into NaN positions and silently drop the star. The
        # distance only enters proper-motion propagation through perspective
        # acceleration, which is below 1e-8 mas over the Gaia to JWST baseline.
        positions = SkyCoord(
            ra=_nan_filled(table["ra"]),
            dec=_nan_filled(table["dec"]),
            pm_ra_cosdec=_nan_filled(table["pmra"]),
            pm_dec=_nan_filled(table["pmdec"]),
            obstime=Time("J2016.0"),
        )
        # A 2-parameter solution has no proper motion and cannot be propagated.
        valid = ~np.isnan(
            [
                positions.ra.value,
                positions.dec.value,
                positions.pm_ra_cosdec.value,
                positions.pm_dec.value,
            ]
        ).any(axis=0)

        new_stars = []
        for id, position, covariance, ok in zip(
            source_id, positions, _astrometric_covariance(table), valid
        ):
            if ok:
                new_stars.append(Star(id, position, covariance))

        return new_stars

    def join_tables(self, tables: list[Table] | None = None) -> Table:
        if tables is None:
            tables = self.data

        non_empty = [t for t in tables if len(t)]
        if len(non_empty) == 0:
            return Table()

        combined = vstack(non_empty, join_type="exact")
        _, unique_idx = np.unique(combined["SOURCE_ID"], return_index=True)
        result = combined[np.sort(unique_idx)]  # restore original order
        return result

    def append(self, data: Table, observation_time: Time | str):
        self.data.append(data)
        self.observation_times.append(observation_time)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Table:
        return self.data[index]

    def __repr__(self):
        return f"StarTable: {len(self.data)}"
