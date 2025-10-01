# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %
from typing import Optional, Union

from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np

from .color import Color
from .parse.grid import GridModel
from .wcs.wcs_astropy import WcsAstropy
from .polarization import PolarizationType


class Grid:
    """
    Grid provides the physical coordinates for a sky brightness model.

    The class holds a spatial coordiante system, which includes
        - `spatial` world coordinate system (WcsAstropy), which locates the
          spatial grid in astrophysical coordinates.
        - `spectral` coordinate system (Color), which provides a spectral
          coordinate range to the frequency/energy/wavelength bins of the sky
          brightness model.
        - `polarization`, for now fixed to Stokes I.
        - `times`, for now fixed to eternity.
    """

    def __init__(
        self,
        spatial: WcsAstropy,
        spectral: Color,
        polarization: Union[tuple, PolarizationType] = ("I",),
    ):
        """
        Initialize the Grid with a `spatial` and `spectral` coordinate system.

        Parameters
        ----------
        spatial : WcsAstropy
            The spatial coordinate system.
        spectral: Color
            The spectral coordinate system.
        """

        assert isinstance(spatial, WcsAstropy)
        assert isinstance(spectral, Color)

        # Spatial
        self.spatial = spatial
        # Spectral
        self.spectral: Color = spectral

        # Polarization
        self.polarization = (
            polarization
            if isinstance(polarization, PolarizationType)
            else PolarizationType(polarization)
        )

        # Time, TODO: Implement more options
        self.times: u.Quantity = u.Unit("s") * np.array([-np.inf, np.inf])

    @classmethod
    def from_shape_and_fov(
        cls,
        spatial_shape: tuple[int, int] | list[int, int],
        fov: u.Quantity,
        frequencies: Optional[u.Quantity] = None,
        sky_center: SkyCoord = SkyCoord(
            ra=np.nan * u.Unit("rad"), dec=np.nan * u.Unit("rad")
        ),
    ) -> "Grid":
        spectral = (
            Color((0.0, np.inf) * u.Unit("Hz"))
            if frequencies is None
            else Color(frequencies)
        )
        return cls(
            spatial=WcsAstropy(center=sky_center, shape=spatial_shape, fov=fov),
            spectral=spectral,
        )

    @classmethod
    def from_grid_model(cls, grid_model: GridModel) -> "Grid":
        """Build Grid from GridModel."""
        spatial = WcsAstropy.from_spatial_model(grid_model.spatial_model)
        spectral = grid_model.color_ranges
        return cls(spatial, spectral)

    @property
    def shape(self):
        """Shape of the grid. (spectral, spatial)"""
        return self.spectral.shape + self.spatial.shape

    def __repr__(self):
        return (
            "Grid("
            f"\npolarization={self.polarization}\n"
            f"\ntimes={self.times}\n"
            f"\nspectral={self.spectral}\n"
            f"\nspatial={self.spatial}\n"
            ")"
        )
