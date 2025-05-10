# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2025 Max-Planck-Society

# %
import jax.numpy as jnp
import nifty8.re as jft
import numpy as np

from ....sky_model.single_correlated_field import build_single_correlated_field


class LearnablePsf(jft.Model):
    def __init__(self, psf_kernel: np.ndarray, modification_model: jft.Model):
        self.static = psf_kernel
        self.modification = modification_model

        self._is_kernel_2d = len(psf_kernel.shape) == 2

        super().__init__(
            domain=(modification_model.domain),
            white_init=True,
        )

    def __call__(self, x):
        kernel = self.static * self.modification(x)
        if self._is_kernel_2d:
            return kernel / jnp.sum(kernel)
        return kernel / kernel.sum(axis=-1).sum(axis=-1)


def build_psf_modification_model_strategy(
    domain_key: str, psf_shape: tuple[int, int, int], strategy: str = "full"
) -> jft.Model:
    assert len(psf_shape) == 3

    if strategy == "full":
        skr = []
        skr_domain = {}
        for ii in range(psf_shape[0]):
            ski, _ = build_single_correlated_field(
                f"{domain_key}_{ii}",
                psf_shape[1:],
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
            return jft.Model(
                lambda x: jnp.array([jnp.exp(sk(x)) for sk in skr]), domain=skr_domain
            )

    if strategy == "single":
        skr, _ = build_single_correlated_field(
            f"{domain_key}",
            psf_shape[1:],
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
        return jft.Model(lambda x: jnp.exp(skr(x)), domain=skr.domain)
