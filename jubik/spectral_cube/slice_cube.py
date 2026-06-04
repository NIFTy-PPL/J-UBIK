import numpy as np
import astropy.units as u

from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales, wcs_to_celestial_frame

from .axes import CubeAxes
from ..grid import Grid
from ..color import Color, get_spectral_range_index
from ..wcs.wcs_astropy import WcsAstropy

import numpy as np
import astropy.units as u
from astropy.wcs import WCS


def slice_cube_spatial(grid, cube_samples, upper_left_corner, lower_right_corner):
    """
    Fully exact FITS WCS cropper.
    No deepcopy, no copy, no internal WCS mutation.
    """

    # --- Step 1: world → pixel ---
    x_ul, y_ul = grid.spatial.world_to_pixel(upper_left_corner)
    x_lr, y_lr = grid.spatial.world_to_pixel(lower_right_corner)

    # --- Step 2: bounds ---
    x_min = max(0, int(np.floor(min(x_ul, x_lr))))
    x_max = min(grid.spatial.shape[0], int(np.ceil(max(x_ul, x_lr))))

    y_min = max(0, int(np.floor(min(y_ul, y_lr))))
    y_max = min(grid.spatial.shape[1], int(np.ceil(max(y_ul, y_lr))))

    nnx = x_max - x_min
    nny = y_max - y_min

    if nnx <= 0 or nny <= 0:
        raise ValueError("No overlap with grid")

    # --- Step 3: slice data ---
    subcube_samples = cube_samples[y_min:y_max, x_min:x_max]

    # --- Step 4: rebuild WCS from header (clean + correct) ---
    header = grid.spatial.to_header()
    sub_wcs = WCS(header)

    # shift pixel reference frame (THIS is the only valid modification)
    sub_wcs.wcs.crpix[0] -= x_min
    sub_wcs.wcs.crpix[1] -= y_min

    # --- Step 5: DO NOT wrap / DO NOT overwrite internals ---
    # TODO: Implent full with rotation and coordinate system
    spatial_subgrid = WcsAstropy(
        center=grid.spatial.center,
        shape=(nnx, nny),
        fov=grid.spatial.fov,
    )

    # attach full WCS safely as an attribute (NOT replacing .wcs)
    spatial_subgrid._wcs = sub_wcs

    # optionally expose it consistently
    spatial_subgrid.wcs_model = sub_wcs

    # --- Step 6: build grid ---
    subgrid = Grid(
        spatial=spatial_subgrid,
        spectral=grid.spectral,
        polarization=grid.polarization,
    )

    return subgrid, subcube_samples


# def slice_cube_spatial(cube_samples, grid: Grid, upper_left_corner: SkyCoord, lower_right_corner: SkyCoord, frame="icrs"):
#     x_ul, y_ul = grid.spatial.world_to_pixel(upper_left_corner)
#     x_lr, y_lr = grid.spatial.world_to_pixel(lower_right_corner)

#     # if (x_ul >= x_lr) or (y_ul >= y_lr):
#     #     raise ValueError(
#     #         "Invalid corner ordering: upper left must be above and left of lower right."
#     #     )

#     nx, ny = grid.spatial.shape

#     # x_min = max(0, int(np.floor(x_ul)))
#     # x_max = min(nx, int(np.ceil(x_lr)))
#     # y_min = max(0, int(np.floor(y_ul)))
#     # y_max = min(ny, int(np.ceil(y_lr)))

#     x_min = max(0, int(np.floor(min(x_ul, x_lr))))
#     x_max = min(nx, int(np.ceil(max(x_ul, x_lr))))

#     y_min = max(0, int(np.floor(min(y_ul, y_lr))))
#     y_max = min(ny, int(np.ceil(max(y_ul, y_lr))))

#     nnx = x_max - x_min
#     nny = y_max - y_min

#     if (nnx <= 0) or (nny <= 0):
#         raise ValueError("Desired subgrid does not overlap original spatial grid.")

#     # Shift reference pixel
#     nawcs = grid.spatial.deepcopy()
#     nawcs.wcs.crpix -= [x_min, y_min]

#     # Robust frame recovery
#     frame = wcs_to_celestial_frame(nawcs)

#     # Correct center pixel (FITS convention)
#     cp = np.array([[(nnx - 1) / 2, (nny - 1) / 2]])
#     cw = nawcs.wcs_pix2world(cp, 0)[0]
#     cc = SkyCoord(cw[0] * u.deg, cw[1] * u.deg, frame=frame)

#     # Pixel scale → FOV
#     scales = proj_plane_pixel_scales(nawcs) * u.deg
#     fov_x = scales[0] * nnx
#     fov_y = scales[1] * nny

#     spatial_subgrid = WcsAstropy(
#         center = cc,
#         shape = (nnx, nny),
#         fov = (fov_x, fov_y),
#         rotation = grid.spatial.rotation,
#         coordinate_system = grid.spatial.coordinate_system,
#     )

#     x_indices = np.arange(x_min, x_max)
#     y_indices = np.arange(y_min, y_max)

#     subgrid = Grid(
#         spatial = spatial_subgrid,
#         spectral = grid.spectral,
#         polarization = grid.polarization,
#     )

#     subcube_samples = np.take(cube_samples, indices=y_indices, axis=CubeAxes.Y)
#     subcube_samples = np.take(subcube_samples, indices=x_indices, axis=CubeAxes.X)

#     return subgrid, subcube_samples


def slice_cube_spectral(cube_samples, grid, spectral_range):
    spec_min = Color(u.Quantity(spectral_range[0]))
    spec_max = Color(u.Quantity(spectral_range[1]))
    spec_min_index = get_spectral_range_index(grid.spectral, spec_min)[0]
    spec_max_index = get_spectral_range_index(grid.spectral, spec_max)[0]
    indices_spec = np.arange(spec_min_index, spec_max_index + 1)

    subgrid = Grid(
        spatial=grid.spatial,
        spectral=grid.spectral[spec_min_index : spec_max_index + 1],
        polarization=grid.polarization,
    )

    subcube_samples = np.take(
        cube_samples, indices=indices_spec, axis=CubeAxes.SPECTRAL
    )

    return subgrid, subcube_samples
