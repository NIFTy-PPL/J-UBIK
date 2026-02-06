from ..wcs.wcs_astropy import WcsAstropy
import numpy as np
import astropy.units as u
from astropy.coordinates import  SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales, wcs_to_celestial_frame

AX_SAMPLE = 0
AX_STOKES = 1
AX_TIME   = 2
AX_SPEC   = 3
AX_Y      = 4
AX_X      = 5
N_AXIS = 6

ID_STOKES_I = 0
ID_STOKES_Q = 1
ID_STOKES_U = 2
ID_STOKES_V = 3

def get_stokes_component(cube_samples: np.ndarray, stokes: str) -> np.ndarray:
    stokes_map = {"I": ID_STOKES_I, "Q": ID_STOKES_Q, "U": ID_STOKES_U, "V": ID_STOKES_V}

    if stokes not in stokes_map:
        raise ValueError(f"Invalid Stokes component '{stokes}'. Must be one of {list(stokes_map.keys())}.")

    idx = stokes_map[stokes]

    if cube_samples.shape[AX_STOKES] <= idx:
        raise ValueError(
            f"Stokes component '{stokes}' (index {idx}) not present in this cube."
        )

    return np.take(cube_samples, [idx], axis=AX_STOKES)

# Subgrid extraction

def create_spatial_subgrid_and_extraction_indices(
    spatial_grid: WcsAstropy,
    upper_left_corner_sky_coord: SkyCoord,
    lower_right_corner_sky_coord: SkyCoord,
):
    x_ul, y_ul = spatial_grid.world_to_pixel(upper_left_corner_sky_coord)
    x_lr, y_lr = spatial_grid.world_to_pixel(lower_right_corner_sky_coord)

    if (x_ul >= x_lr) or (y_ul >= y_lr):
        raise ValueError(
            "Invalid corner ordering: upper left must be above and left of lower right."
        )

    nx, ny = spatial_grid.shape

    x_min = max(0, int(np.floor(x_ul)))
    x_max = min(nx, int(np.ceil(x_lr)))
    y_min = max(0, int(np.floor(y_ul)))
    y_max = min(ny, int(np.ceil(y_lr)))

    nnx = x_max - x_min
    nny = y_max - y_min

    if (nnx <= 0) or (nny <= 0):
        raise ValueError("Desired subgrid does not overlap original spatial grid.")

    # Shift reference pixel
    nawcs = spatial_grid.deepcopy()
    nawcs.wcs.crpix -= [x_min, y_min]

    # Robust frame recovery
    frame = wcs_to_celestial_frame(nawcs)

    # Correct center pixel (FITS convention)
    cp = np.array([[(nnx - 1) / 2, (nny - 1) / 2]])
    cw = nawcs.wcs_pix2world(cp, 0)[0]
    cc = SkyCoord(cw[0] * u.deg, cw[1] * u.deg, frame=frame)

    # Pixel scale → FOV
    scales = proj_plane_pixel_scales(nawcs) * u.deg
    fov_x = scales[0] * nnx
    fov_y = scales[1] * nny

    sub_spatial_grid = WcsAstropy(
        center=cc,
        shape=(nnx, nny),
        fov=(fov_x, fov_y),
        rotation= spatial_grid.rotation,
        coordinate_system=spatial_grid.coordinate_system,
    )
    
    x_indices = np.arange(x_min, x_max)
    y_indices = np.arange(y_min, y_max)
    
    return sub_spatial_grid, x_indices, y_indices

def broadcast_one_dim_to_full(array: np.ndarray, axis: int) -> np.ndarray:
    shape = [1] * N_AXIS
    shape[axis] = array.size
    return array.reshape(shape)


# Cube operations

def sum_cube(cube, axis):
    return np.sum(cube, axis=axis, keepdims=True)

def spectral_normalized_cube(cube):
    norm = sum_cube(cube, AX_SPEC)
    return cube/norm

def integrate_cube(cube, axis, delta):
    return delta*sum_cube(cube, axis)

def velocity_map(cube, velocities, spectral_norm_cube=None):
    # Computes the intensity weighted frist velocity mean for each pixel
    vel = broadcast_one_dim_to_full(velocities, AX_SPEC)

    if spectral_norm_cube is None:
        norm_cube = spectral_normalized_cube(cube)
    else:
        # Used only if normalized cube was computed beforehand to reduce redundancy operations
        norm_cube = spectral_norm_cube

    return np.sum(norm_cube*vel, axis=AX_SPEC, keepdims=True)

def velocity_centered_moment_map_statistical(cube, velocities, order):
    # Computes a moment N map as defined by statistical centered moments
    # order = 2: Velocity dispersion
    # order = 3: Proportional to skewness of intensity weighted velocity distribution
    if order < 2:
        raise ValueError("Only statistical cumulative moment of second or higher order can be calculated.")
    norm_cube = spectral_normalized_cube(cube)
    vel = broadcast_one_dim_to_full(velocities, AX_SPEC)
    vel_mean = velocity_map(None, velocities, norm_cube) # Cube has not to be set as norm cube was computed already
    centered_vel = vel - vel_mean

    return np.sum(norm_cube*centered_vel**order, axis=AX_SPEC, keepdims=True)

def cube_to_fits(cube):
    return cube.cube_samples, cube.flux_density_unit, ""

def cube_spectral_integrated(cube) -> tuple[np.ndarray, u.Unit, str]:
    delta, unit = cube.spectral_bin_width # frequency
    field = integrate_cube(cube.cube_samples, AX_SPEC, delta)
    current_unit = cube.flux_density_unit*unit
    file_name = "spectral_integrated_cube"

    return field, current_unit, file_name

def cube_spatial_integrated(cube) -> tuple[np.ndarray, u.Unit, str]:
    delta, unit = cube.spatial_pixel_area # area
    field = integrate_cube(cube.cube_samples, (AX_Y, AX_X), delta)
    current_unit = cube.flux_density_unit*unit
    file_name = "spatial_integrated_cube"

    return field, current_unit, file_name

def cube_spatial_spectral_integrated(cube) -> tuple[np.ndarray, u.Unit, str]:
    delta, unit = cube.cube_voxel_volume
    field = integrate_cube(cube.cube_samples, (AX_SPEC, AX_Y, AX_X), delta)
    current_unit = cube.flux_density_unit*unit
    file_name = "spatial_spectral_integrated_cube"

    return field, current_unit, file_name

def cube_spectral_averaged(cube) -> tuple[np.ndarray, u.Unit, str]:
    spec_int_cube, current_unit, _ = cube_spectral_integrated(cube)
    spec_vol = cube.grid.spectral.max() - cube.grid.spectral.min()
    field = spec_int_cube/spec_vol.value
    file_name = "spectral_averaged_cube"

    return field, current_unit, file_name

def cube_spatial_averaged(cube) -> tuple[np.ndarray, u.Unit, str]:
    spat_int_cube, current_unit, _ = cube_spatial_integrated(cube)
    spat_vol = cube.grid.spatial.fov[0]*cube.grid.spatial.fov[1]
    field = spat_int_cube/spat_vol.value
    file_name = "spatial_averaged_cube"

    return field, current_unit, file_name

def cube_spatial_spectral_averaged(cube) -> tuple[np.ndarray, u.Unit, str]:
    spat_int_cube, current_unit, _ = cube_spatial_integrated(cube)
    spec_vol = cube.grid.spectral.max() - cube.grid.spectral.min()
    spat_vol = cube.grid.spatial.fov[0]*cube.grid.spatial.fov[1]
    spec_spat_vol = spec_vol*spat_vol
    field = spat_int_cube/spec_spat_vol.value
    file_name = "spatial_spectral_averaged_cube"

    return field, current_unit, file_name

def cube_stokesI_weighted_velocity_mean(cube) -> tuple[np.ndarray, u.Unit, str]:
    vel, current_unit = cube.velocities
    field = velocity_map(cube.stokesI, vel)
    file_name = f"stokesI_weighted_velocity_mean_reference_{str(cube.reference_frequency).replace(' ', '')}"

    return field, current_unit, file_name

def cube_stokesI_weighted_velocity_std(cube) -> tuple[np.ndarray, u.Unit, str]:
    vel, current_unit = cube.velocities
    field = np.sqrt(velocity_centered_moment_map_statistical(cube.stokesI,vel,2))
    file_name = f"stokesI_weighted_velocity_std_reference_{str(cube.reference_frequency).replace(' ', '')}"

    return field, current_unit, file_name

def cube_stokesI_weighted_velocity_variance(cube) -> tuple[np.ndarray, u.Unit, str]:
    vel, current_unit = cube.velocities
    field = velocity_centered_moment_map_statistical(cube.stokesI,vel,2)
    file_name = f"stokesI_weighted_velocity_variance_reference_{str(cube.reference_frequency).replace(' ', '')}"

    return field, current_unit**2, file_name

def cube_stokesI_weighted_velocity_skewness(cube) -> tuple[np.ndarray, u.Unit, str]:
    vel, _ = cube.velocities

    cm2 = velocity_centered_moment_map_statistical(cube.stokesI,vel,2)
    cm3 = velocity_centered_moment_map_statistical(cube.stokesI,vel,3)

    field = cm3/(cm2**(3/2))
    current_unit = u.dimensionless_unscaled
    file_name = f"stokesI_weighted_velocity_skewness_reference_{str(cube.reference_frequency).replace(' ', '')}"

    return field, current_unit, file_name

def cube_stokesI_weighted_velocity_excess_kurtosis(cube) -> tuple[np.ndarray, u.Unit, str]:
    vel, _ = cube.velocities

    cm2 = velocity_centered_moment_map_statistical(cube.stokesI,vel,2)
    cm4 = velocity_centered_moment_map_statistical(cube.stokesI,vel,4)
    
    field = cm4/cm2**2 - 3
    current_unit = u.dimensionless_unscaled
    file_name = f"stokesI_weighted_velocity_excess_kurtosis_reference_{str(cube.reference_frequency).replace(' ', '')}"

    return field, current_unit, file_name

def cube_linear_polarization_intensity(cube) -> tuple[np.ndarray, u.Unit, str]:
    current_unit = cube.flux_density_unit
    field = np.sqrt(cube.stokesQ**2 + cube.stokesU**2)
    file_name = "linear_polarization_intensity"

    return field, current_unit, file_name

def cube_fractional_linear_polarization(cube) -> tuple[np.ndarray, u.Unit, str]:
    current_unit = u.dimensionless_unscaled
    field = np.sqrt(cube.stokesQ**2 + cube.stokesU**2)/cube.stokesI
    file_name = "linear_polarization_intensity"

    return field, current_unit, file_name

def cube_polarization_angle(cube) -> tuple[np.ndarray, u.Unit, str]:
    current_unit = u.rad
    field = 0.5*np.arctan2(cube.stokesU, cube.stokesQ)
    file_name = "polarization_angle"

    return field, current_unit, file_name

def cube_circular_polarization_fraction(cube) -> tuple[np.ndarray, u.Unit, str]:
    current_unit = u.dimensionless_unscaled
    field = cube.stokesV/cube.stokesI
    file_name = "circular_polarization_fraction"

    return field, current_unit, file_name

def cube_total_polarized_intensity(cube) -> tuple[np.ndarray, u.Unit, str]:
    current_unit = cube.flux_density_unit
    field = np.sqrt(cube.stokesQ**2 + cube.stokesU**2 + cube.stokesV**2)
    file_name = "total_polarized_intensity"

    return field, current_unit, file_name