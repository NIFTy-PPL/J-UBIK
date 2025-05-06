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


def plot_test(filter_alignment, ii, jwst_data):
    import astropy
    import matplotlib.pyplot as plt

    stars = filter_alignment.get_stars(ii)
    plot_position_stars = partial(
        plot_sky_coords,
        sky_coords=[star.position for star in stars],
        labels=[star.id for star in stars],
        behavior_index=lambda index, sky_coords: (
            sky_coords if index == 0 else [sky_coords[index - 1]]
        ),
    )
    jwst_wcs = astropy.wcs.WCS(jwst_data.wcs.to_header())

    shape = (
        int(
            (
                filter_alignment.alignment_meta.fov
                / jwst_data.meta.pixel_distance.to(
                    filter_alignment.alignment_meta.fov.unit
                )
            ).value
        ),
    ) * 2

    data = [jwst_data.dm.data]
    wcs = [jwst_wcs]
    for star in stars:
        minx, maxx, miny, maxy = star.bounding_indices(jwst_data, shape)
        print(star.id, minx, maxx, miny, maxy)
        wcs.append(jwst_wcs[miny : maxy + 1, minx : maxx + 1])
        data.append(jwst_data.data_inside_extrema((minx, maxx, miny, maxy)))

    mean = np.nanmean(jwst_data.dm.data)
    fig, axs = plot_jwst_panels(
        data,
        wcs,
        nrows=1,
        ncols=len(data),
        vmin=0.9 * mean,
        vmax=1.1 * mean,
        coords_plotter=plot_position_stars,
    )


