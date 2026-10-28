import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from jubik.instruments.jwst.alignment.star_alignment import Star
from jubik.instruments.jwst.alignment.star_model import build_star_position_offset
from jubik.instruments.jwst.data.loader.stars_loader import _shift_jacobian
from jubik.models.distributions import build_parametric_prior_from_prior_config
from jubik.parse.models.distributions import prior_config_factory
from jubik.instruments.jwst.parse.rotation_and_shift.coordinates_correction import (
    CoordinatesCorrectionPriorConfig,
)
from jubik.instruments.jwst.rotation_and_shift.coordinates_correction import (
    ShiftAndRotationCorrection,
    build_coordinates_corrected_for_field,
    build_coordinates_corrected_for_stars,
)
from jubik.wcs.wcs_astropy import WcsAstropy

CENTER = SkyCoord(10 * u.deg, -5 * u.deg)
SOURCE = CENTER.spherical_offsets_by(2 * u.arcsec, -3 * u.arcsec)
SIGMA = 1.5  # arcsec, source width
N_DATA = 31
GRID = WcsAstropy(CENTER, shape=41, fov=41 * 0.5 * u.arcsec)
# Star position offset model whose value at latent `e` is `e` arcsec.
POSITION = build_parametric_prior_from_prior_config(
    "star_0_position", prior_config_factory(["normal", 0.0, 1.0]), (2,), as_model=True
)


def _correction(shifts):
    """Shared shift model whose value at latent `shifts` is `shifts` arcsec."""
    prior = CoordinatesCorrectionPriorConfig.from_yaml_dict(
        dict(
            model="shift",
            shift=["normal", 0.0, 1.0],
            shift_unit="arcsec",
            rotation=["normal", 0.0, 1.0],
            rotation_unit="deg",
        ),
        shift_shape=shifts.shape,
        rotation_shape=(shifts.shape[0], 1),
    )
    correction = ShiftAndRotationCorrection("f", prior, CENTER)
    params = {"f_shift": shifts, "f_rotation": np.zeros((shifts.shape[0], 1))}
    assert np.allclose(correction.shift(params), shifts)
    return correction, params


def _target_centroids(data_wcs, shifts):
    """Data (x, y) centroid of the source as seen by the target response."""
    xx, yy = np.meshgrid(np.arange(N_DATA), np.arange(N_DATA), indexing="xy")
    centers = data_wcs.pixel_to_world(xx, yy)
    correction, params = _correction(shifts)
    coords = build_coordinates_corrected_for_field(
        correction, GRID, [centers] * shifts.shape[0]
    )(params)
    # Evaluate the sky analytically at the corrected sampling positions.
    src = np.array(GRID.world_to_indices_yx(SOURCE))[:, None, None]
    scale = GRID.pixel_scales_yx.to_value(u.arcsec)[:, None, None]
    r2 = (((np.asarray(coords) - src) * scale) ** 2).sum(axis=1)
    image = np.exp(-0.5 * r2 / SIGMA**2)
    norm = image.sum(axis=(1, 2))
    return np.stack(
        [(image * xx).sum(axis=(1, 2)) / norm, (image * yy).sum(axis=(1, 2)) / norm],
        axis=1,
    )


def _star_positions(data_wcs, shifts, observation_ids, epsilon=None):
    star = Star(0, SOURCE)
    p0 = np.array(star.pixel_position_in_subsampled_data(data_wcs))
    jac = _shift_jacobian(star, data_wcs, GRID, 0, 0, 1)
    n = len(observation_ids)
    correction, params = _correction(shifts)
    position = None if epsilon is None else POSITION
    model = build_coordinates_corrected_for_stars(
        correction,
        np.tile(p0, (n, 1)),
        np.tile(jac, (n, 1, 1)),
        observation_ids,
        position,
    )
    if epsilon is None:
        assert "star_0_position" not in model.domain
        return model(params)
    assert model.domain["star_0_position"].shape == (2,)
    return model(params | {"star_0_position": epsilon})


@pytest.mark.parametrize(
    "position_angle, pixel_scales",
    [(0, (1.0, 1.0)), (0, (0.7, 1.3)), (30, (1.0, 1.0)), (30, (0.7, 1.3))],
)
def test_shared_shift_moves_star_and_target_alike(position_angle, pixel_scales):
    data_wcs = WcsAstropy(
        CENTER,
        shape=N_DATA,
        fov=[s * N_DATA for s in pixel_scales] * u.arcsec,
        position_angle=position_angle * u.deg,
    )
    shifts = np.array([[0.3, -0.6], [-0.8, 0.5]])
    zero = np.zeros_like(shifts)

    target = _target_centroids(data_wcs, shifts) - _target_centroids(data_wcs, zero)
    # The star is only in the second exposure.
    star = _star_positions(data_wcs, shifts, [1]) - _star_positions(data_wcs, zero, [1])

    assert np.abs(target[1]).max() > 0.3
    np.testing.assert_allclose(star[0], target[1], atol=1e-4)


def test_aligned_shift_moves_star_against_sampling_rows_and_columns():
    # Issue #141 table: aligned grids, 1 arcsec pixels. Shift [1, 0] moves the
    # target one row up, [0, 1] one column left; the star must do the same.
    data_wcs = WcsAstropy(CENTER, shape=N_DATA, fov=N_DATA * u.arcsec)
    shifts = np.array([[1.0, 0.0], [0.0, 1.0]])
    star = _star_positions(data_wcs, shifts, [0, 1]) - _star_positions(
        data_wcs, np.zeros_like(shifts), [0, 1]
    )
    np.testing.assert_allclose(star, [[0.0, -1.0], [-1.0, 0.0]], atol=1e-6)


def test_star_position_moves_star_like_the_opposite_shift():
    # A star truly at Gaia + e on the grid appears where a shared shift of -e
    # puts its Gaia position, in every exposure.
    data_wcs = WcsAstropy(
        CENTER,
        shape=N_DATA,
        fov=[0.7 * N_DATA, 1.3 * N_DATA] * u.arcsec,
        position_angle=30 * u.deg,
    )
    e = np.array([0.4, -0.7])
    star = _star_positions(data_wcs, np.zeros((2, 2)), [0, 1], epsilon=e)

    np.testing.assert_allclose(
        star, _star_positions(data_wcs, np.tile(-e, (2, 1)), [0, 1]), atol=1e-10
    )
    scales = GRID.pixel_scales_yx.to_value(u.arcsec)
    indices = GRID.world_to_indices_yx(SOURCE) + e / scales
    true = Star.skycoord_to_subpixel(GRID.indices_yx_to_world(*indices), data_wcs)
    np.testing.assert_allclose(star, [true, true], atol=1e-6)


def test_star_position_without_shift_correction_raises():
    with pytest.raises(ValueError, match="needs a shift correction"):
        build_coordinates_corrected_for_stars(
            None, np.zeros((1, 2)), np.zeros((1, 2, 2)), [0], POSITION
        )


@pytest.mark.parametrize("position_angle", [0.0, 30.0, 90.0] * u.deg)
def test_star_position_offset_carries_the_gaia_covariance_to_the_grid(position_angle):
    # Map the offsets of the two unit latents back to the sky: their outer product
    # must give the Gaia covariance along (RA cos(dec), Dec), for any grid
    # rotation and with the column axis running against RA.
    grid = WcsAstropy(
        CENTER,
        shape=41,
        fov=[0.4 * 41, 0.6 * 41] * u.arcsec,
        position_angle=position_angle,
    )
    covariance = np.array([[0.04, 0.03], [0.03, 0.09]])  # arcsec^2
    model = build_star_position_offset(Star(0, SOURCE, covariance), grid)
    scales = grid.pixel_scales_yx.to_value(u.arcsec)
    indices = np.array(grid.world_to_indices_yx(SOURCE))
    columns = []
    for xi in np.eye(2):
        offset = np.asarray(model({"star_0_position": xi}))
        moved = grid.indices_yx_to_world(*(indices + offset / scales))
        offsets = SOURCE.spherical_offsets_to(moved)
        columns.append([o.to_value(u.arcsec) for o in offsets])
    sky = np.array(columns).T
    np.testing.assert_allclose(sky @ sky.T, covariance, rtol=1e-5, atol=1e-8)
