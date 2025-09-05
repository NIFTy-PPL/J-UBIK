import jubik0 as ju

import astropy.units as u

from src.parse.wcs import sky_center


# NOTE: This is a small introduction to the Grid class.
# We will look at different ways how to initialize them and some features.
# 1. Initialization
#   1.1 Simplest only spatial information
#   1.2 Simplest spatial with pointing information
#   1.3 Simplest with frequency information
#   1.4 From grid model ( from config.yaml )


# 1. Initialization

# 1.1. Most simplest, no center information
grid = ju.Grid.from_shape_and_fov(spatial_shape=(128, 128), fov=(1.0, 1.0) * u.arcmin)

# Inspect the grid:
print(grid.shape)
print(grid.spatial.shape)
print(grid.spectral.shape)


# 1.2. Simple with spatial center
# NOTE: THis provides more advanced interpolation features
from astropy.coordinates import SkyCoord

center = SkyCoord(ra=12 * u.rad, dec=77 * u.deg)
grid = ju.Grid.from_shape_and_fov(
    spatial_shape=(128, 128),
    fov=(1.0, 1.0) * u.arcmin,
    sky_center=center,
)

print(grid.spatial.center)


# 1.3 Simple with spectral information
grid = ju.Grid.from_shape_and_fov(
    spatial_shape=(128, 128),
    fov=(1.0, 1.0) * u.arcmin,
    frequencies=[  # Three energy bins can be irregular and with gaps.
        (12 * u.eV, 13 * u.eV),  # start, end
        (13 * u.eV, 14 * u.eV),
        (15 * u.eV, 18 * u.eV),
    ],
    # NOTE: Also different unit system works.
    # frequencies=[
    #     (12 * u.Hz, 13 * u.Hz),
    #     (13 * u.Hz, 14 * u.Hz),
    #     (15 * u.Hz, 18 * u.Hz),
    # ],
)

print(grid.shape)


# 1.4. From GridModel
# NOTE: This version meant to interface with the yaml config file
from jubik0.parse.grid import GridModel

gm = GridModel.from_yaml_dict(
    {
        "sdim": 256,
        "s_padding_ratio": 1.5,
        "fov": "0.5arcmin",
        "rotation": "0deg",
        "coordinate_frame": "icrs",
        "sky_center": {"ra": "175.20125deg", "dec": "-26.48583333deg"},
        "energy_unit": "eV",
        "energy_bin": {
            "e_min": [0.00058436571],
            "e_max": [0.0037682094],
            "reference_bin": 0,
        },
    }
)

grid = ju.Grid.from_grid_model(gm)
