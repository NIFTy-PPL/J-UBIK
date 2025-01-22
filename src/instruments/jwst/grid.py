# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %

import numpy as np
from typing import Union

from .wcs.wcs_astropy import WcsAstropy
from .color import ColorRange, ColorRanges

from .parse.grid import GridModel


class Grid:
    """
    Grid provides the physical coordinates for a sky brightness model.

    The class holds a spatial coordiante system, which includes
        - `spatial` world coordinate system (WcsAstropy), which locates the
          spatial grid in astrophysical coordinates.
        - `spectral` coordinate system (ColorRanges), which provides a spectral
          coordinate range to the frequency/energy/wavelength bins of the sky
          brightness model.
        - `polarization_labels`, for now fixed to Stokes I.
        - `times`, for now fixed to eternity.
    """

    def __init__(
        self,
        spatial: WcsAstropy,
        spectral: Union[ColorRange, ColorRanges],
    ):
        """
        Initialize the Grid with a `spatial` and `spectral` coordinate system.

        Parameters
        ----------
        spatial : WcsAstropy
            The spatial coordinate system.
        spectral: ColorRanges
            The spectral coordinate system.
        """

        # Spatial
        self.spatial = spatial
        # Spectral
        self.spectral = spectral
        # Polarization, TODO: Implement more options.
        self.polarization_labels = ['I']
        # Time, TODO: Implement more options
        self.times = [-np.inf, np.inf]

    @classmethod
    def from_grid_model(cls, grid_model: GridModel):
        '''Build Grid from GridModel.'''
        spatial = WcsAstropy.from_wcs_model(grid_model.wcs_model)
        spectral = grid_model.color_ranges
        return Grid(spatial, spectral)

    @property
    def shape(self):
        '''Shape of the grid. (spectral, spatial)'''
        return self.spectral.shape + self.spatial.shape

    def __repr__(self):
        return ('Grid('
                f'\npolarization_labels={self.polarization_labels}\n'
                f'\ntimes={self.times}\n'
                f'\nspectral={self.spectral}\n'
                f'\nspatial={self.spatial}\n'
                ')')
