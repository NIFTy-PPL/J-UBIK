import jax.numpy as jnp
import nifty.re as jft
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from jax import lax, vmap, Array


from ....models.distributions import build_parametric_prior_from_prior_config
from ....wcs.wcs_astropy import WcsAstropy

from .star_alignment import Star
from ..data.loader.stars_loader import StarsData
from ....parse.models.distributions import ProbabilityConfig
from ..rotation_and_shift.coordinates_correction import (
    Coordinates,
    CoordinatesCorrectedStarShift,
    ShiftAndRotationCorrection,
    build_coordinates_corrected_for_stars,
)


def bilinear_point_source_evaluation(
    sky: np.ndarray | Array,
    location_xy: np.ndarray | Array,
    flux: float | np.ndarray | Array,
):
    """Calculate the point source in sky using bilinear interpolation weights.

    Parameters
    ----------
    sky: np.ndarray | jax.Array
        The sky to which the point source is added.
    location_xy: np.ndarray
        The location of the point source. Note:
        1) Assuming the astropy convention that the center of the reference pixel has
           the index (0, 0). Hence the upper-left corner (in `ij` indexing) of the
           reference pixel has the index (-0.5, -0.5).
        2) Assuming `xy` indexing of the location, hence x-position of the location is
           in the first index of the location array, and the y-position in the second
           index of the location, i.e.: (x, y).
           Note: The sky array is `ij` indexed, hence the first index tells the
           y-position. Hence, we have to revert the pixel position, see `pixel_pos`.
           Furthermore, the bilinear weights are evaluated accordingly:
           w00           w01
              +----|-x--+
              |    |    |
              |    |    |
              +---------+
              y    | .  |
              |    |    |
              +----|----+
           w10           w11
    flux: float | np.ndarray | Array
        The flux value of the point source.

    Warning
    -------
    The implementation breaks down for locations outside the grid spanned by the center
    of the sky pixels. Hence, all negative locations, e.g. (-0.5, -0.5), and all
    locations beyond (shape-1, shape-1), lead to erroneous evaluations of the flux.
    """
    loc_in_pixel, pixel_pos = jnp.modf(location_xy)
    pixel_pos = pixel_pos.astype(int)[::-1]

    x, y = loc_in_pixel

    # bilinear weights
    w00 = (1.0 - x) * (1.0 - y)
    w01 = x * (1.0 - y)
    w10 = (1.0 - x) * y
    w11 = x * y

    point_source = jnp.array(((w00, w01), (w10, w11)))
    point_source = flux * point_source

    # Indexing into array
    sky_slice = lax.dynamic_slice(sky, pixel_pos, point_source.shape)
    return lax.dynamic_update_slice(sky, sky_slice + point_source, pixel_pos)


class StarInData(jft.Model):
    def __init__(
        self,
        skies: np.ndarray,
        brightness: jft.Model,
        location: Coordinates | CoordinatesCorrectedStarShift,
    ):
        self._skies = skies
        self.brightness = brightness
        self.location = location
        self._vmap_call = vmap(bilinear_point_source_evaluation, (0, 0, None))

        super().__init__(domain=brightness.domain | location.domain)

    def __call__(self, x):
        brightness = self.brightness(x)
        locations = self.location(x)
        return self._vmap_call(self._skies, locations, brightness)


def _grid_to_sky(grid_wcs: WcsAstropy, position: SkyCoord) -> np.ndarray:
    """Sky offset (RA cos(dec), Dec) per offset along the grid (row, column) at
    `position`, both in arcsec, shape (2, 2).

    Inverts central differences of `world_to_indices_yx` along the sky, which
    converts `position` into the grid frame."""
    step = 0.1 * u.arcsec
    sky_to_grid = np.stack(
        [
            np.subtract(
                *(
                    grid_wcs.world_to_indices_yx(
                        position.spherical_offsets_by(*(sign * d))
                    )
                    for sign in (1, -1)
                )
            )
            for d in np.eye(2) * step
        ],
        axis=1,
    ) * (grid_wcs.pixel_scales_yx.to_value(u.arcsec)[:, None] / (2 * step.value))
    return np.linalg.inv(sky_to_grid)


def _star_position_offset(star: Star, times: list[Time | str]) -> jft.Model:
    """Offset of the star from its Gaia position in each exposure.

    The star's astrometry (position and proper motion) deviates from Gaia's by
    `cholesky(star.astrometric_covariance) @ latent`. The model propagates that
    deviation to each of `times`.

    Parameters
    ----------
    star: Star
        The Gaia star. Needs its `astrometric_covariance`.
    times: list[Time | str]
        The epoch of each exposure the star is in.

    Returns
    -------
    jft.Model
        Maps the standard normal latent `star_<id>_astrometry`, shape (4,), to the
        true minus Gaia position along (RA cos(dec), Dec) in arcsec, shape
        (len(times), 2). The latent has no filter prefix, so all filters share it.
    """
    covariance = star.astrometric_covariance
    if covariance is None or np.isnan(covariance).any():
        raise ValueError(
            f"Gaia star {star.id} has no astrometric covariance: an error or "
            "correlation of ra, dec, pmra or pmdec is missing, masked or NaN "
            "(a star table cached without error columns needs a fresh "
            "query). Exclude it with `exclude_source_id` or turn off "
            "`fit_star_positions`."
        )

    cholesky = np.linalg.cholesky(covariance) * u.mas.to(u.arcsec)
    latent_to_offset = np.stack([star.epoch_jacobian(t) @ cholesky for t in times])
    key = f"star_{star.id}_astrometry"
    return jft.Model(
        lambda x: latent_to_offset @ x[key], domain={key: jft.ShapeWithDtype((4,))}
    )


def build_star_in_data(
    filter_key: str,
    star: Star,
    star_light_prior: ProbabilityConfig,
    star_data: StarsData,
    shift_and_rotation_correction: ShiftAndRotationCorrection | None,
    grid_wcs: WcsAstropy,
    observation_times: list[Time | str],
    fit_position: bool = False,
) -> StarInData:
    """Build star in data field.

    Parameters
    ----------
    filter_key: str
        Name of the filter
    star: Star
        The Gaia star. Its id names the brightness prior.
    star_light_prior: ProbabilityConfig
        The prior for the star brightness
    star_data: StarsData
        The cutouts of the star, one per exposure in `star_data.observation_ids`.
    shift_and_rotation_correction: ShiftAndRotationCorrection | None
        The shift and rotation of each exposure.
    grid_wcs: WcsAstropy
        The sky grid. The exposure shifts are along its (row, column) axes.
    observation_times: list[Time | str]
        The epoch of every exposure, indexed by `star_data.observation_ids`.
    fit_position: bool
        Fit the star's offset from its Gaia position, with the Gaia covariance as
        prior. Default False.
    """

    skies = np.array(star_data.sky_array)

    brightness = build_parametric_prior_from_prior_config(
        f"{filter_key}_{star.id}_brightness",
        star_light_prior,
        shape=(),
        as_model=True,
    )

    times = [observation_times[i] for i in star_data.observation_ids]
    sky_to_data = star_data.sky_to_data.to_value(u.pix / u.arcsec)
    shift_jacobian = np.stack(
        [
            j @ _grid_to_sky(grid_wcs, star.position_at(t))
            for j, t in zip(sky_to_data, times)
        ]
    )
    position = _star_position_offset(star, times) if fit_position else None

    location_of_star_in_data_subpixels = build_coordinates_corrected_for_stars(
        shift_and_rotation_correction=shift_and_rotation_correction,
        pixel_coordinates=np.array(star_data.star_in_subsampled_pixels),
        shift_jacobian=shift_jacobian,
        observation_ids=star_data.observation_ids,
        position=position,
        position_jacobian=sky_to_data,
    )

    return StarInData(skies, brightness, location_of_star_in_data_subpixels)
