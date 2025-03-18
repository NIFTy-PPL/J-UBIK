from ..parse.sky_model.resolve_diffuse import (
    ResolveDiffuseSkyModel, ResolveSingleFrequencyStokesIDiffuseModel)
from .single_correlated_field import build_single_correlated_field

import nifty8.re as jft
from jax import numpy as jnp

from dataclasses import asdict


class DiffuseModel(jft.Model):
    def __init__(
        self,
        prefix: str,
        out_shape: tuple[int, int, int, int, int],
        log_diffuse: jft.Model
    ):
        self._prefix = prefix
        self._out_shape = out_shape
        self.log_diffuse = log_diffuse
        super().__init__(domain={prefix: log_diffuse.domain})

    def __call__(self, x):
        return jnp.broadcast_to(
            jnp.exp(self.log_diffuse(x[self._prefix])), self._out_shape)


def sky_model_diffuse(
    sky_shape: tuple[int, int],
    sky_distances: tuple[float, float],
    resolve_diffuse_sky_model: ResolveDiffuseSkyModel
):
    '''Builds the DiffuseModel from the ResolveDiffuseSkyModel.

    Parameters
    ----------
    sky_shape: tuple[int, int]
        The shape of the sky.
    sky_distances: tuple[float, float]
        The distances of the sky.
    resolve_diffuse_sky_model: ResolveDiffuseSkyModel
        The config for the diffuse model.

    Returns
    ------
    DiffuseModel
    '''

    if isinstance(resolve_diffuse_sky_model.diffuse_config,
                  ResolveSingleFrequencyStokesIDiffuseModel):
        bg_log_diffuse, additional = build_single_correlated_field(
            prefix=resolve_diffuse_sky_model.prefix,
            shape=sky_shape,
            distances=sky_distances,
            zero_mode_config=asdict(
                resolve_diffuse_sky_model.diffuse_config.zero_mode_model),
            fluctuations_config=asdict(
                resolve_diffuse_sky_model.diffuse_config.fluctuations_model),
        )
        bg_diffuse_model = DiffuseModel(
            prefix=resolve_diffuse_sky_model.prefix,
            out_shape=(1, 1, 1) + sky_shape,
            log_diffuse=bg_log_diffuse
        )
        return bg_diffuse_model, additional

    raise NotImplementedError
