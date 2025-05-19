import jax.numpy as jnp
import nifty8.re as jft
import numpy as np
from jax import lax, vmap, Array


from ..parametric_model.parametric_prior import build_parametric_prior_from_prior_config

from ..data.jwst_data import DataMetaInformation
from ..data.loading.stars_loader import StarData
from ..parse.parametric_model.parametric_prior import ProbabilityConfig
from ..rotation_and_shift.coordinates_correction import (
    Coordinates,
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
        location: Coordinates | ShiftAndRotationCorrection,
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


class StarAndSky(jft.Model):
    def __init__(
        self,
        star_id: int,
        skies: np.ndarray,
        brightness: jft.Model,
        location: Coordinates | ShiftAndRotationCorrection,
    ):
        from ....sky_model.single_correlated_field import build_single_correlated_field

        # self._skies = skies
        skr = []
        skr_domain = {}
        for ii in range(skies.shape[0]):
            ski, _ = build_single_correlated_field(
                f"{star_id}_{ii}",
                skies.shape[1:],
                distances=(1.0, 1.0),
                zero_mode_config=dict(
                    offset_mean=0.0,
                    offset_std=(1.0, 2.0),
                ),
                fluctuations_config=dict(
                    fluctuations=(1.0, 5e-1),
                    loglogavgslope=(-5.0, 2e-1),
                    flexibility=(1e0, 2e-1),
                    asperity=(5e-1, 5e-2),
                    non_parametric_kind="power",
                ),
            )
            skr.append(ski)
            skr_domain = skr_domain | ski.domain
        self.skies = jft.Model(
            lambda x: jnp.array([jnp.exp(sk(x)) for sk in skr]), domain=skr_domain
        )

        self.brightness = brightness
        self.location = location
        self._vmap_call = vmap(bilinear_point_source_evaluation, (0, 0, None))

        super().__init__(domain=brightness.domain | location.domain | self.skies.domain)

    def __call__(self, x):
        brightness = self.brightness(x)
        locations = self.location(x)
        skies = self.skies(x)
        return self._vmap_call(skies, locations, brightness)


def build_star_in_data(
    filter_key: str,
    filter_meta: DataMetaInformation,
    star_id: int,
    star_light_prior: ProbabilityConfig,
    star_data: StarData,
    shift_and_rotation_correction: ShiftAndRotationCorrection | None,
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
        pixel_distance=filter_meta.pixel_distance / star_data.subsample,
        observation_ids=star_data.observation_ids,
        shift_only=True,
    )

    # return StarAndSky(star_id, skies, brightness, location_of_star_in_data_subpixels)
    return StarInData(skies, brightness, location_of_star_in_data_subpixels)
