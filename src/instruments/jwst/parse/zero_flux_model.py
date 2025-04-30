# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2024 Max-Planck-Society

# %
from .parametric_model.parametric_prior import (
    ProbabilityConfig,
    prior_config_factory,
)

from dataclasses import dataclass

DEFAULT_KEY = "default"


@dataclass
class ZeroFluxPriorConfigs:
    default: ProbabilityConfig
    names: dict[str, ProbabilityConfig]

    def get_name_setting_or_default(self, filter_name: str) -> ProbabilityConfig:
        """Returns the PriorConfig for the `filter_name` or the default.

        Parameters
        ----------
        filter_name : str
            The filter in question.
        """
        filter_name = filter_name.lower()
        if filter_name in self.names:
            return self.names[filter_name]
        return self.default


def yaml_to_zero_flux_prior_config(zero_flux_config: dict | None):
    if zero_flux_config is None:
        return None

    default = prior_config_factory(zero_flux_config[DEFAULT_KEY])

    names = {}
    for filter_name, filter_prior in zero_flux_config.items():
        filter_name = filter_name.lower()
        if filter_name == DEFAULT_KEY:
            continue

        names[filter_name] = prior_config_factory(filter_prior)

    return ZeroFluxPriorConfigs(default=default, names=names)
