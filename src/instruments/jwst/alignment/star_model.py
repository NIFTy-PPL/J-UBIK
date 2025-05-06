import jax
import jax.numpy as jnp
import nifty8.re as jft
import numpy as np
from jax import lax, vmap

from ..parametric_model.parametric_prior import build_parametric_prior_from_prior_config
from ..parse.jwst_likelihoods import StarData
from ..parse.parametric_model.parametric_prior import ProbabilityConfig
from .star_alignment import Star
from ..rotation_and_shift.coordinates_correction import (
    Coordinates,
    ShiftAndRotationCorrection,
    build_coordinates_corrected_for_stars,
)


def square_point_source_in_sky_from_location(sky, location, flux):
    loc_in_pixel, pixel_pos = jnp.modf(location[::-1] - jnp.array([0.5, 0.5]))
    pixel_pos = pixel_pos.astype(int)

    y, x = loc_in_pixel

    # bilinear weights
    w00 = (1.0 - x) * (1.0 - y)
    w01 = x * (1.0 - y)
    w10 = (1.0 - x) * y
    w11 = x * y

    point_source = jnp.array(((w00, w01), (w10, w11)))
    point_source = flux * point_source

    sky_slice = lax.dynamic_slice(sky, pixel_pos, point_source.shape)
    sky = lax.dynamic_update_slice(sky, sky_slice + point_source, pixel_pos)
    return sky


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
        self._call = vmap(square_point_source_in_sky_from_location, (0, 0, None))

        super().__init__(domain=brightness.domain | location.domain)

    def __call__(self, x):
        brightness = self.brightness(x)
        locations = self.location(x)
        return self._call(self._skies, locations, brightness)


def build_star_in_data(
    filter_key: str,
    star: Star,
    star_light_prior: ProbabilityConfig,
    star_data: StarData,
    shift_and_rotation_correction: ShiftAndRotationCorrection,
):
    skies = np.array(star_data.sky_array)
    data_meta = star_data.meta

    brightness = build_parametric_prior_from_prior_config(
        f"{filter_key}_{star.id}_brightness",
        star_light_prior,
        shape=(),
        as_model=True,
    )

    location_of_star_in_data_subpixels = build_coordinates_corrected_for_stars(
        shift_and_rotation_correction=shift_and_rotation_correction,
        pixel_coordinates=np.array(star_data.star_in_subsampled_pixels),
        pixel_distance=data_meta.pixel_distance / data_meta.subsample,
        observation_ids=star_data.observation_ids,
        shift_only=True,
    )

    return StarInData(skies, brightness, location_of_star_in_data_subpixels)


if __name__ == "__main__":
    import numpy as np
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from ..plotting.plotting_sky import plot_sky_coords, plot_jwst_panels
    from functools import partial
    from astropy.wcs import WCS
    import matplotlib.pyplot as plt

    position = np.array((0.0, 0.5))  # centre of pixel (1,0)

    pix_scale = 1.0 * u.arcsec  # desired pixel size
    deg_per_pix = pix_scale.to(u.deg).value  # ≃ 2.777…×10⁻⁴ deg

    # -- data already given -------------------------------------------------
    pix_zero = SkyCoord(ra=11.0 * u.deg, dec=41.0 * u.deg, frame="icrs")

    # -- 4×4 WCS with 1″ pixels --------------------------------------------
    N = 4
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1, 1]  # centre pixel (0, 0)
    wcs.wcs.cdelt = [-deg_per_pix, deg_per_pix]  # RA decreases to the right
    wcs.wcs.crval = [pix_zero.ra.deg, pix_zero.dec.deg]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]  # gnomonic projection

    # convert that pixel to RA/Dec with the same WCS
    star_sc = wcs.pixel_to_world(*position)
    x_star, y_star = wcs.world_to_pixel(star_sc)  # [[2]]
    print(x_star, y_star)  # → (2.5, 2.5)  (centre of image)

    # pixel centres
    yy, xx = np.mgrid[:N, :N] + 0.5
    coords_small = wcs.pixel_to_world(xx, yy)  # 4×4 SkyCoord

    data = np.arange(N * N).reshape(N, N) / 16
    sky = np.zeros((N, N), dtype=float)
    star_inter = square_point_source_in_sky_from_location(sky, position + 0.5, 1.0)

    star_position_plot = partial(plot_sky_coords, sky_coords=[star_sc])
    fig, axes = plot_jwst_panels(
        [data, star_inter],
        [wcs, wcs],
        vmax=data.max(),
        nrows=1,
        ncols=2,
        coords_plotter=star_position_plot,
    )
    axes[0].scatter(*position)
    plt.show()

# flux = 1.0
