import jax.numpy as jnp
import nifty.re as jft
import numpy as np
from astropy import units as u
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


def build_star_position_offset(
    star: Star, grid_wcs: WcsAstropy, times: list[Time | str]
) -> jft.Model:
    """The star's true minus Gaia position at each of `times` in arcsec along the
    grid (row, column), shape (len(times), 2), see
    `build_coordinates_corrected_for_stars`.

    Prior: the Gaia `astrometric_covariance` on the star's position and proper
    motion. `epoch_jacobian` carries it to each epoch and the local
    (RA cos(dec), Dec) to grid Jacobian into the grid frame, so the correlations
    between epochs, a rotated grid and the Gaia correlations come out right. The
    latent `star_<id>_astrometry`, shape (4,), is standard normal and has no filter
    prefix, so every filter shares it.
    """

    def sky_to_grid(position):
        row_column = np.array(grid_wcs.world_to_indices_yx(position))
        return (
            np.stack(
                [
                    np.array(
                        grid_wcs.world_to_indices_yx(position.spherical_offsets_by(*d))
                    )
                    - row_column
                    for d in ((1, 0) * u.arcsec, (0, 1) * u.arcsec)
                ],
                axis=1,
            )
            * grid_wcs.pixel_scales_yx.to_value(u.arcsec)[:, None]
        )

    cholesky = np.linalg.cholesky(star.astrometric_covariance) * u.mas.to(u.arcsec)
    latent_to_offset = np.stack(
        [
            sky_to_grid(star.position_at(t)) @ star.epoch_jacobian(t) @ cholesky
            for t in times
        ]
    )
    key = f"star_{star.id}_astrometry"
    return jft.Model(
        lambda x: latent_to_offset @ x[key], domain={key: jft.ShapeWithDtype((4,))}
    )


def build_star_in_data(
    filter_key: str,
    star_id: int,
    star_light_prior: ProbabilityConfig,
    star_data: StarsData,
    shift_and_rotation_correction: ShiftAndRotationCorrection | None,
    position: jft.Model | None = None,
) -> StarInData:
    """Build star in data field.

    Parameters
    ----------
    filter_key: str
        Name of the filter
    star_id: int
        The identifcation number of the star, used for the brightness prior.
    star_light_prior: ProbabilityConfig
        The prior for the star brightness
    position: jft.Model | None
        The star's true minus Gaia position, see
        `build_coordinates_corrected_for_stars`.
    """

    skies = np.array(star_data.sky_array)

    brightness = build_parametric_prior_from_prior_config(
        f"{filter_key}_{star_id}_brightness",
        star_light_prior,
        shape=(),
        as_model=True,
    )

    location_of_star_in_data_subpixels = build_coordinates_corrected_for_stars(
        shift_and_rotation_correction=shift_and_rotation_correction,
        pixel_coordinates=np.array(star_data.star_in_subsampled_pixels),
        shift_jacobian=star_data.shift_jacobian,
        observation_ids=star_data.observation_ids,
        position=position,
    )

    return StarInData(skies, brightness, location_of_star_in_data_subpixels)
