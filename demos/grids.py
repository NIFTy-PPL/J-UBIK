import jubik as ju

import astropy.units as u


# NOTE: This is a small introduction to the Grid class.
# We will look at different ways how to initialize them and some features.
# 1. Initialization
#   1.1 Simplest only spatial information
#   1.2 Simplest spatial with pointing information
#   1.3 Simplest with frequency information
#   1.4 From grid model ( from config.yaml )


# 1. Initialization

# 1.1. Most simplest, no center information
grid = ju.Grid.from_shape_and_fov(shape=(128, 128), fov=(1.0, 1.0) * u.arcmin)

# Inspect the grid:
print(grid.shape)
print(grid.spatial.shape_xy)
print(grid.spectral.shape)


# 1.2. Simple with spatial center
# NOTE: THis provides more advanced interpolation features
from astropy.coordinates import SkyCoord

center = SkyCoord(ra=12 * u.rad, dec=77 * u.deg)
grid = ju.Grid.from_shape_and_fov(
    shape=(128, 128),
    fov=(1.0, 1.0) * u.arcmin,
    sky_center=center,
)

print(grid.spatial.center)


# 1.3 Simple with spectral information
grid = ju.Grid.from_shape_and_fov(
    shape=(128, 128),
    fov=(1.0, 1.0) * u.arcmin,
    frequencies=[  # Three energy bins can be irregular and with gaps.
        (12, 13),  # start, end
        (13, 14),
        (15, 18),
    ]
    * u.eV,
    # NOTE: Also different unit system works.
    # frequencies=[(12, 13), (13, 14), (15, 18)] * u.Hz,
)

print(grid.shape)


# 1.4. From GridModel
# NOTE: This version meant to interface with the yaml config file
from jubik.parse.grid import GridModel

gm = GridModel.from_yaml_dict(
    {
        "shape": 256,
        "s_padding_ratio": 1.5,
        "fov": "0.5arcmin",
        "position_angle": "0deg",
        "frame": "icrs",
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


# 2. Plot sky coordinates, including a non-zero position angle.
# WCSAxes uses the full celestial transform; a rectangular `extent` cannot
# represent a rotated grid. Both panels show the same YX array, never sky.T.
import matplotlib.pyplot as plt
import numpy as np
from jubik.wcs import SpatialGeometry, WcsAstropy


def plot_sky_coordinates():
    """Return a reproducible random sky on unrotated and rotated WCS axes."""
    center = SkyCoord(ra=10 * u.deg, dec=-5 * u.deg, frame="icrs")
    geometry = SpatialGeometry.from_xy((96, 64), (48, 32) * u.arcsec)
    sky_yx = np.random.default_rng(42).normal(size=geometry.shape_yx)
    fig = plt.figure(figsize=(11, 4.5), layout="constrained")
    for panel, angle in enumerate((0, 30), start=1):
        wcs = WcsAstropy.from_geometry(geometry, center, angle * u.deg)
        ax = fig.add_subplot(1, 2, panel, projection=wcs)
        image = ax.imshow(sky_yx, origin="lower", vmin=-3, vmax=3, cmap="viridis")
        for coordinate in ax.coords:
            coordinate.set_format_unit(u.deg)
            coordinate.set_major_formatter("d.dddd")
            coordinate.set_ticklabel(simplify=False, exclude_overlapping=True)
        ax.coords[0].set_axislabel("Right ascension (ICRS)")
        ax.coords[1].set_axislabel("Declination (ICRS)")
        ax.coords.grid(color="white", alpha=0.5, linestyle=":")
        ax.set_title(f"Position angle = {angle}°")
    fig.colorbar(image, ax=fig.axes, label="Random sky brightness (arbitrary units)")
    return fig


if __name__ == "__main__":
    plot_sky_coordinates()
    plt.show()
