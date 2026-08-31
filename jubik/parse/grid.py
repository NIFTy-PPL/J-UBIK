from .color import yaml_to_binned_colors
from .wcs.spatial_model import SpatialModel
from ..color import Color


from dataclasses import dataclass


@dataclass
class GridModel:
    spatial_model: SpatialModel
    color_ranges: Color

    @classmethod
    def from_yaml_dict(cls, grid_config: dict):
        """
        Builds the reconstruction grid from the given configuration.

        The reconstruction grid is defined by the world location, field of view
        (FOV), shape (resolution), and rotation, all specified in the input
        configuration. These parameters are extracted from the grid_config dictionary
        using helper functions.

        Parameters
        ----------
        grid_config : dict
            - `sky_center`: dict[str: str]
                e.g.: {ra: '0deg', dec: '1deg'}
            - `fov`: str | tuple[str]
                e.g.: 0.5arcsec; [0.1arcmin, 2.0deg]
            - `sdim`: tuple[int, int]
                Shape of the grid, i.e. resolution, as (sdim, sdim).
                e.g.: [128, 12]
            - `rotation`: str,
                Rotation of the grid.
                e.g.: 0.1deg
            - `energy_bin`: Holding `e_min`, `e_max`, and `reference_bin`.
                e.g.: [e_min: [0.1], e_max: [1.2]]
            - `energy_unit`: The units for `e_min` and `e_max`
                e.g.: eV

        Returns:
        --------
        GridModel
            The GridModel which holds
                - spatial_model: how to build the wcs for the spatial coordinates.
                - color_ranges: The Color for the energies.
                - color_reference_bin: The reference_bin for the energy model.
        """
        spatial_model = SpatialModel.from_yaml_dict(grid_config)
        color_ranges = yaml_to_binned_colors(grid_config)

        return GridModel(
            spatial_model=spatial_model,
            color_ranges=color_ranges,
        )
