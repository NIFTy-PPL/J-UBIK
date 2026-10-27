import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from jubik.instruments.jwst.alignment.star_alignment import Star
from jubik.instruments.jwst.data.loader.stars_loader import _shift_jacobian
from jubik.instruments.jwst.parse.rotation_and_shift.coordinates_correction import (
    CoordinatesCorrectionPriorConfig,
)
from jubik.instruments.jwst.rotation_and_shift.coordinates_correction import (
    OFFSET_DOMAIN_KEY,
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


def _correction(shifts, offset=None):
    """Shared shift model whose value at latent `shifts` is `shifts` arcsec.

    With `offset`, the model carries the shared offset prior too, at latent
    `offset` (also `offset` arcsec).
    """
    raw = dict(
        model="shift",
        shift=["normal", 0.0, 1.0],
        shift_unit="arcsec",
        rotation=["normal", 0.0, 1.0],
        rotation_unit="deg",
    )
    if offset is not None:
        raw["offset"] = ["normal", 0.0, 1.0]
    prior = CoordinatesCorrectionPriorConfig.from_yaml_dict(
        raw, shift_shape=shifts.shape, rotation_shape=(shifts.shape[0], 1)
    )
    correction = ShiftAndRotationCorrection("f", prior, CENTER)
    params = {"f_shift": shifts, "f_rotation": np.zeros((shifts.shape[0], 1))}
    if offset is not None:
        params[OFFSET_DOMAIN_KEY] = np.asarray(offset)
        assert np.allclose(correction.field_shift(params), shifts + offset)
    assert np.allclose(correction.shift(params), shifts)
    return correction, params


def _target_centroids(data_wcs, shifts, offset=None):
    """Data (x, y) centroid of the source as seen by the target response."""
    xx, yy = np.meshgrid(np.arange(N_DATA), np.arange(N_DATA), indexing="xy")
    centers = data_wcs.pixel_to_world(xx, yy)
    correction, params = _correction(shifts, offset)
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


def _star_positions(data_wcs, shifts, observation_ids, offset=None):
    star = Star(0, SOURCE)
    p0 = np.array(star.pixel_position_in_subsampled_data(data_wcs))
    jac = _shift_jacobian(star, data_wcs, GRID, 0, 0, 1)
    n = len(observation_ids)
    correction, params = _correction(shifts, offset)
    return build_coordinates_corrected_for_stars(
        correction, np.tile(p0, (n, 1)), np.tile(jac, (n, 1, 1)), observation_ids
    )(params)


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


def test_offset_moves_the_target_like_a_shift_and_the_star_not_at_all():
    data_wcs = WcsAstropy(CENTER, shape=N_DATA, fov=N_DATA * u.arcsec)
    shifts = np.array([[0.3, -0.6], [-0.8, 0.5]])
    offset = np.array([0.4, -0.7])

    with_offset = _target_centroids(data_wcs, shifts, offset)
    as_shift = _target_centroids(data_wcs, shifts + offset)
    np.testing.assert_allclose(with_offset, as_shift, atol=1e-6)

    star = _star_positions(data_wcs, shifts, [0, 1], offset)
    np.testing.assert_allclose(star, _star_positions(data_wcs, shifts, [0, 1]))

    correction, _ = _correction(shifts, offset)
    assert OFFSET_DOMAIN_KEY in correction.field_shift_domain
    plain, _ = _correction(shifts)
    assert plain.offset is None and OFFSET_DOMAIN_KEY not in plain.domain
