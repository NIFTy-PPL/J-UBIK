import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from ..reconstruction_grid import Grid


def coords(shape: int, distance: float):
    '''Returns coordinates such that the edge of the array is
    shape/2*distance'''
    halfside = shape/2 * distance
    return np.linspace(-halfside+distance/2, halfside-distance/2, shape)


def get_xycoords(shape: tuple[int], distances: tuple[float]):
    assert len(shape) == 2
    if np.isscalar(distances):
        distances = (float(distances),) * len(shape)
    x_direction = coords(shape[0], distances[0])
    y_direction = coords(shape[1], distances[1])
    return np.array(np.meshgrid(x_direction, y_direction, indexing='xy'))


def rotation_(grid, theta):
    """
    Rotates the coordinate grid anticlockwise by an angle theta
    """
    x = grid[0] * np.cos(theta) - grid[1] * np.sin(theta)
    y = grid[0] * np.sin(theta) + grid[1] * np.cos(theta)
    return x, y


CENTER = SkyCoord(0*u.rad, 0*u.rad)


def test_coords():
    shape = (128,)*2
    fov = (128*0.05*u.arcsec,)*2
    distances = (0.05,)*2

    coor = get_xycoords(shape, distances)
    grid = Grid(CENTER, shape=shape, fov=fov)

    assert np.allclose(coor, grid.rel_coords())

    # The extended array is rolled, as we use the circular bla
    shape_extended = (128+64,)*2
    coor_ext = get_xycoords(shape_extended, distances)
    coor_ext = np.roll(coor_ext, (-32, -32), axis=(1, 2))

    assert np.allclose(coor_ext, grid.rel_coords(extend_factor=1.5))


def test_rotation():
    '''Check that roation by 25 degrees inside Grid returns the expected
    rotated coordinates.'''

    shape = (128, 128)
    fov = (128*0.05*u.arcsec,)*2
    distances = (0.05,)*2
    rotation = 25*u.deg

    coor = get_xycoords(shape, distances)
    grid = Grid(CENTER, shape=shape, fov=fov, rotation=rotation)

    xy_rotated = grid.rel_coords()
    xy_rotated_old = np.array(rotation_(coor, rotation.to(u.rad).value))
    assert np.allclose(xy_rotated, xy_rotated_old)


def test_interpolation_points():
    big_shape = (512, )*2
    sma_shape = (128, )*2

    big_fov = (25*u.arcsec,)*2
    sma_fov = (10*u.arcsec,)*2
    rotation = 25*u.deg

    big_grid = Grid(CENTER, shape=big_shape, fov=big_fov)
    sma_grid = Grid(CENTER, shape=sma_shape, fov=sma_fov, rotation=rotation)

    xy_rotated = sma_grid.rel_coords()
    interpolation_test = (
        (xy_rotated - big_grid.rel_coords()[0, 0, 0]) /
        big_grid.distances[0].to(u.arcsec).value
    )
    interpolation = big_grid.wcs.index_from_wl(sma_grid.wl_coords())
    interpolation = interpolation[0]

    assert np.allclose(interpolation_test, interpolation)
