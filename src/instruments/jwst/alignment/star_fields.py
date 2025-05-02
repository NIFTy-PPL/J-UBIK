import jax.numpy as jnp
import nifty8.re as jft

from ..parametric_model.parametric_prior import build_parametric_prior_from_prior_config
from ..parse.parametric_model.parametric_prior import ProbabilityConfig


class Stars(jft.Model):
    def __init__(
        self,
        filter_name: str,
        shape: tuple[int, int],
        star_ids: tuple[int],
        brightness: jft.Model,
    ):
        for sh in shape:
            assert sh % 2 != 0, "Provide uneven shape for stars"
        self.out = jnp.zeros(brightness.target.shape + tuple(shape))
        self.position = [shp // 2 + 1 for shp in shape]
        self.brightness = brightness
        self._target_ids = [f"{filter_name}_{id}" for id in star_ids]

        super().__init__(domain=brightness.domain)

    def __call__(self, x):
        out_arr = self.out.at[..., self.position[0], self.position[1]].set(
            self.brightness(x)
        )
        # return out_arr
        return {tid: out_arr[ii] for ii, tid in enumerate(self._target_ids)}


def build_stars(
    domain_key: str,
    shape: tuple[int, int],
    star_ids: tuple[int],
    star_light_prior: ProbabilityConfig,
):
    for sh in shape:
        assert sh % 2 != 0, "Provide uneven shape for stars"

    brightness = build_parametric_prior_from_prior_config(
        f"{domain_key}_star_brightness",
        star_light_prior,
        shape=len(star_ids),
        as_model=True,
    )

    return Stars(domain_key, shape, star_ids, brightness)
