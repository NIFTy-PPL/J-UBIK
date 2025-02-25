from .color import (
    yaml_to_binned_colors, yaml_to_color_reference_bin,
    cfg_to_binned_colors, cfg_to_color_reference_bin
)
from .wcs.spatial_model import SpatialModel
from ..color import ColorRanges


from configparser import ConfigParser
from dataclasses import dataclass


@dataclass
class GridModel:
    spatial_model: SpatialModel
    color_ranges: ColorRanges
    color_reference_bin: int = 0

    @classmethod
    def from_yaml_dict(cls, grid_config: dict):
        '''
        Builds the reconstruction grid from the given configuration.

        The reconstruction grid is defined by the world location, field of view
        (FOV), shape (resolution), and rotation, all specified in the input
        configuration. These parameters are extracted from the grid_config dictionary
        using helper functions.

        Parameters
        ----------
        grid_config : dict
            The configuration dictionary containing the following keys:
            - `sky_center`: World coordinate of the spatial grid center.
            - `fov`: Field of view of the grid in appropriate units.
            - `sdim`: Shape of the grid, i.e. resolution, as (sdim, sdim).
            - `rotation`: Rotation of the grid.
            - `energy_bin`: Holding `e_min`, `e_max`, and `reference_bin`.
            - `energy_unit`: The units for `e_min` and `e_max`

        Returns:
        --------
        GridModel
            The GridModel which holds
                - spatial_model: how to build the wcs for the spatial coordinates.
                - color_ranges: The ColorRanges for the energies.
                - color_reference_bin: The reference_bin for the energy model.
        '''
        spatial_model = SpatialModel.from_yaml_dict(grid_config)
        color_ranges = yaml_to_binned_colors(grid_config)
        color_reference_bin = yaml_to_color_reference_bin(grid_config)

        return GridModel(
            spatial_model=spatial_model,
            color_ranges=color_ranges,
            color_reference_bin=color_reference_bin
        )

    @classmethod
    def from_config_parser(cls, grid_config: ConfigParser):
        '''
        Builds the reconstruction grid from the given configuration.

        The reconstruction grid is defined by the world location, field of view
        (FOV), shape (resolution), and rotation, all specified in the input
        configuration. These parameters are extracted from the grid_config dictionary
        using helper functions.

        Parameters
        ----------
        grid_config : dict
            The configuration dictionary containing the following keys:
            - `sky_center`: World coordinate of the spatial grid center.
            - `fov`: Field of view of the grid in appropriate units.
            - `fov_unit`: (Optional) unit for the fov.
            - `sdim`: Shape of the grid, i.e. resolution, as (sdim, sdim).
            - `rotation`: Rotation of the grid.
            - `rotation_unit`: (Optional) unit for the rotation.
            - `energy_bin`: Holding `e_min`, `e_max`, and `reference_bin`.
            - `energy_unit`: The units for `e_min` and `e_max`

        Returns:
        --------
        GridModel
            The GridModel which holds
                - spatial_model: how to build the wcs for the spatial coordinates.
                - color_ranges: The ColorRanges for the energies.
                - color_reference_bin: The reference_bin for the energy model.
        '''

        spatial_model = SpatialModel.from_config_parser(grid_config)
        color_ranges = cfg_to_binned_colors(grid_config)
        color_reference_bin = cfg_to_color_reference_bin(grid_config)

        return GridModel(
            spatial_model=spatial_model,
            color_ranges=color_ranges,
            color_reference_bin=color_reference_bin
        )
