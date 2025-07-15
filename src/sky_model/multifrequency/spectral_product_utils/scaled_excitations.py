# Copyright(C) 2025
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Matteo Guardiani & Julian Rüstig

from typing import Union

from nifty.re.model import Model
from nifty.re.num.stats_distributions import lognormal_prior, normal_prior

from .distribution_or_default import build_distribution_or_default


class ScaledExcitations(Model):
    def __init__(self, fluctuations: Model, latent_space_xi: Model):
        self.fluctuations = fluctuations
        self.latent_space_xi = latent_space_xi
        super().__init__(domain=fluctuations.domain | latent_space_xi.domain)

    def __call__(self, p):
        return self.fluctuations(p) * self.latent_space_xi(p)


def build_scaled_excitations(
    prefix: str,
    fluctuations_settings: Union[callable, tuple, list],
    shape: tuple[int],
) -> ScaledExcitations:
    fluctuations = build_distribution_or_default(
        fluctuations_settings, f"{prefix}_fluctuations", lognormal_prior
    )
    latent_space_xi = build_distribution_or_default(
        (0.0, 1.0), f"{prefix}_xi", normal_prior, shape=shape
    )
    return ScaledExcitations(fluctuations, latent_space_xi)
