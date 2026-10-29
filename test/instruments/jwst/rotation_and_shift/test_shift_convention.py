from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from jubik.instruments.jwst.alignment.star_alignment import Star
from jubik.instruments.jwst.alignment.star_model import (
    _grid_to_sky,
    _star_position_offset,
    build_star_in_data,
)
from jubik.instruments.jwst.data.loader.stars_loader import _sky_to_data
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
    p0 = np.array(Star.skycoord_to_subpixel(SOURCE, data_wcs))
    sky = _sky_to_data(SOURCE, data_wcs, 1).to_value(u.pix / u.arcsec)
    jac = sky @ _grid_to_sky(GRID, SOURCE)
    n = len(observation_ids)
    correction, params = _correction(shifts)
    position = None if epsilon is None else POSITION
    model = build_coordinates_corrected_for_stars(
        correction,
        np.tile(p0, (n, 1)),
        np.tile(jac, (n, 1, 1)),
        observation_ids,
        position,
        np.tile(sky, (n, 1, 1)),
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


def test_star_position_moves_star_to_its_true_sky_position():
    # A star truly at Gaia + e along (RA cos(dec), Dec) appears where the data wcs
    # puts that sky position, in every exposure.
    data_wcs = WcsAstropy(
        CENTER,
        shape=N_DATA,
        fov=[0.7 * N_DATA, 1.3 * N_DATA] * u.arcsec,
        position_angle=30 * u.deg,
    )
    e = np.array([0.4, -0.7])
    star = _star_positions(data_wcs, np.zeros((2, 2)), [0, 1], epsilon=e)

    true = Star.skycoord_to_subpixel(
        SOURCE.spherical_offsets_by(*e * u.arcsec), data_wcs
    )
    np.testing.assert_allclose(star, [true, true], atol=1e-4)


def test_sky_to_data_at_the_detector_corner():
    # A bounded gwcs like JWST's (NaN off the detector), 0.031 arcsec pixels,
    # rotated, the star 0.05 pixels inside two opposite corners: half of each
    # difference falls off the detector.
    pytest.importorskip("gwcs")
    from astropy import coordinates
    from astropy.modeling import models
    from gwcs import coordinate_frames, wcs

    from jubik.instruments.jwst.wcs_jwst_data import WcsJwstData

    scale, rotation = 0.031, 30.0  # arcsec, deg
    center = (N_DATA - 1) / 2
    detector_to_sky = (
        (models.Shift(-center) & models.Shift(-center))
        | (models.Scale(-scale / 3600) & models.Scale(scale / 3600))
        | models.Rotation2D(rotation)
        | models.Pix2Sky_TAN()
        | models.RotateNative2Celestial(CENTER.ra.deg, CENTER.dec.deg, 180)
    )
    gwcs = wcs.WCS(
        [
            (
                coordinate_frames.Frame2D(
                    name="detector", axes_names=("x", "y"), unit=(u.pix, u.pix)
                ),
                detector_to_sky,
            ),
            (
                coordinate_frames.CelestialFrame(
                    reference_frame=coordinates.ICRS(), unit=(u.deg, u.deg)
                ),
                None,
            ),
        ]
    )
    gwcs.bounding_box = ((-0.5, N_DATA - 0.5), (-0.5, N_DATA - 0.5))
    data_wcs = WcsJwstData(gwcs)
    assert np.isnan(data_wcs.pixel_to_world(-1.0, -1.0).ra)

    # (RA cos(dec), Dec) per pixel is Rotation2D @ Scale; the Jacobian inverts it.
    theta = np.deg2rad(rotation)
    sky_per_pixel = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    ) @ np.diag([-scale, scale])
    for corner in [(-0.45, -0.45), (N_DATA - 0.55, N_DATA - 0.55)]:
        star = data_wcs.pixel_to_world(*corner)
        np.testing.assert_allclose(
            _sky_to_data(star, data_wcs, 2).to_value(u.pix / u.arcsec),
            2 * np.linalg.inv(sky_per_pixel),
            rtol=1e-4,
        )


def test_star_position_without_shift_correction_or_jacobian_raises():
    with pytest.raises(ValueError, match="needs a shift correction"):
        build_coordinates_corrected_for_stars(
            None, np.zeros((1, 2)), np.zeros((1, 2, 2)), [0], POSITION
        )
    correction, _ = _correction(np.zeros((1, 2)))
    with pytest.raises(ValueError, match="needs its `position_jacobian`"):
        build_coordinates_corrected_for_stars(
            correction, np.zeros((1, 2)), np.zeros((1, 2, 2)), [0], POSITION
        )


def test_star_position_offset_carries_the_gaia_covariance_to_the_epochs():
    # The offsets of the four unit latents at two epochs, as an outer product, must
    # give the Gaia covariance propagated to both epochs along (RA cos(dec), Dec),
    # including the covariance between the epochs.
    std = np.array([200.0, 300.0, 100.0, 50.0])  # mas, mas/yr
    correlation = np.full((4, 4), 0.3) + 0.7 * np.eye(4)
    star = Star(
        0,
        SkyCoord(
            SOURCE.ra,
            SOURCE.dec,
            pm_ra_cosdec=0 * u.mas / u.yr,
            pm_dec=0 * u.mas / u.yr,
            obstime=Time("J2016.0"),
        ),
        std[:, None] * correlation * std[None, :],
    )
    times = [Time("J2016.0") + dt * u.yr for dt in (2, 5)]
    model = _star_position_offset(star, times)
    sky = np.array(
        [np.ravel(model({"star_0_astrometry": xi})) for xi in np.eye(4)]
    ).T * u.arcsec.to(u.mas)
    jacobian = np.vstack([star.epoch_jacobian(t) for t in times])
    np.testing.assert_allclose(
        sky @ sky.T,
        jacobian @ star.astrometric_covariance @ jacobian.T,
        rtol=1e-5,
        atol=1e-3,
    )


def test_star_in_data_does_not_depend_on_the_other_exposures():
    # The same star, shift and epoch must land on the same pixel in filters with
    # different exposure lists. A moving star near the pole, where the grid axes
    # turn along its path, would catch a grid Jacobian or offset frozen at the
    # first exposure.
    star = Star(
        0,
        SkyCoord(
            10 * u.deg,
            89.999 * u.deg,
            pm_ra_cosdec=100 * u.mas / u.yr,
            pm_dec=-30 * u.mas / u.yr,
            obstime=Time("J2016.0"),
        ),
        np.diag([1.0, 4.0, 0.16, 0.16]),  # mas^2, (mas/yr)^2
    )
    times = [Time("2022-01-01"), Time("2024-01-01")]
    grid = WcsAstropy(star.position_at(times[1]), shape=41, fov=20 * u.arcsec)
    correction, params = _correction(np.array([[0.0, 0.0], [1.0, -0.5]]))

    def location(observation_ids):
        n = len(observation_ids)
        star_data = SimpleNamespace(
            sky_array=np.zeros((n, 3, 3)),
            star_in_subsampled_pixels=np.ones((n, 2)),
            sky_to_data=np.tile(np.eye(2), (n, 1, 1)) * u.pix / u.arcsec,
            observation_ids=np.array(observation_ids),
        )
        model = build_star_in_data(
            "f",
            star,
            prior_config_factory(["normal", 1.0, 0.1]),
            star_data,
            correction,
            grid,
            times,
            fit_position=True,
        ).location
        return model(params | {"star_0_astrometry": np.array([1.0, 0, 0, 1.0])})

    np.testing.assert_allclose(location([0, 1])[1], location([1])[0], atol=1e-9)
