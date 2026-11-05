"""The `rshift` coordinates correction: a rotation about a centre, then the shift (#169)."""

import jax
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from jubik.instruments.jwst.parse.rotation_and_shift.coordinates_correction import (
    CoordinatesCorrectionPriorConfig,
)
from jubik.instruments.jwst.rotation_and_shift.coordinates_correction import (
    ShiftAndRotationCorrection,
    build_coordinates_corrected_for_field,
)
from jubik.wcs.wcs_astropy import WcsAstropy

CENTER = SkyCoord(10 * u.deg, -5 * u.deg)
OFF_CENTER = CENTER.spherical_offsets_by(2 * u.arcsec, -3 * u.arcsec)
SHIFTS = np.array([[0.3, -0.6], [-0.8, 0.5]])  # arcsec, (row, column)


def _correction(model, center):
    """Shift in arcsec and rotation in deg equal to their latents."""
    prior = CoordinatesCorrectionPriorConfig.from_yaml_dict(
        dict(
            model=model,
            shift=["normal", 0.0, 1.0],
            shift_unit="arcsec",
            rotation=["normal", 0.0, 1.0],
            rotation_unit="deg",
        ),
        shift_shape=(2, 2),
        rotation_shape=(2, 1),
    )
    return ShiftAndRotationCorrection("f", prior, center)


def _sampling():
    data_wcs = WcsAstropy(
        OFF_CENTER, shape=9, fov=9 * 0.4 * u.arcsec, position_angle=17 * u.deg
    )
    xx, yy = np.meshgrid(np.arange(9), np.arange(9), indexing="xy")
    world = data_wcs.pixel_to_world(xx, yy)
    return world, [world, world]


def _explicit(grid, world, center, theta_deg, shift):
    """Each position turned by theta from North through East about the centre on
    the sphere, then moved by the shift along the grid rows and columns."""
    moved = center.directional_offset_by(
        center.position_angle(world) + theta_deg * u.deg, center.separation(world)
    )
    rows, columns = grid.world_to_indices_yx(moved)
    scale = grid.pixel_scales_yx.to_value(u.arcsec)
    return np.array([rows + shift[0] / scale[0], columns + shift[1] / scale[1]])


GRIDS = [
    WcsAstropy(CENTER, shape=41, fov=41 * 0.5 * u.arcsec),
    WcsAstropy(
        CENTER,
        shape=(41, 31),
        fov=(41 * 0.4, 31 * 0.6) * u.arcsec,
        position_angle=30 * u.deg,
    ),
]


@pytest.mark.parametrize("grid", GRIDS, ids=["square", "rotated-anisotropic"])
@pytest.mark.parametrize("theta", [(-0.5, 0.05), (3.0, 30.0)])
@pytest.mark.parametrize("per_exposure_center", [False, True])
@pytest.mark.parametrize("center", [CENTER, OFF_CENTER], ids=["grid", "off"])
def test_equals_explicit_rotation_about_the_centre_plus_shift(
    grid, theta, per_exposure_center, center
):
    centers = [center, OFF_CENTER] if per_exposure_center else [center, center]
    rotation_center = SkyCoord(centers) if per_exposure_center else center
    world, worlds = _sampling()
    coordinates = build_coordinates_corrected_for_field(
        _correction("rshift", rotation_center), grid, worlds
    )
    got = coordinates({"f_shift": SHIFTS, "f_rotation": np.array(theta)[:, None]})
    for i in range(2):
        np.testing.assert_allclose(
            got[i], _explicit(grid, world, centers[i], theta[i], SHIFTS[i]), atol=1e-6
        )


@pytest.mark.parametrize("grid", GRIDS, ids=["square", "rotated-anisotropic"])
def test_zero_rotation_reproduces_shift_exactly(grid):
    _, worlds = _sampling()
    shift_only, rshift = (
        build_coordinates_corrected_for_field(_correction(m, OFF_CENTER), grid, worlds)
        for m in ("shift", "rshift")
    )
    assert set(shift_only.domain) == {"f_shift"}
    assert set(rshift.domain) == {"f_shift", "f_rotation"}
    params = {"f_shift": SHIFTS, "f_rotation": np.zeros((2, 1))}
    np.testing.assert_array_equal(rshift(params), shift_only(params))


def test_jit_and_grad():
    grid = GRIDS[1]
    _, worlds = _sampling()
    coordinates = build_coordinates_corrected_for_field(
        _correction("rshift", OFF_CENTER), grid, worlds
    )
    params = {"f_shift": SHIFTS, "f_rotation": np.array([[0.5], [-2.0]])}
    np.testing.assert_allclose(
        jax.jit(coordinates)(params), coordinates(params), rtol=1e-12
    )
    gradient = jax.grad(lambda p: coordinates(p).sum())(params)
    assert np.all(np.isfinite(gradient["f_rotation"]))
    assert np.all(gradient["f_rotation"] != 0)
