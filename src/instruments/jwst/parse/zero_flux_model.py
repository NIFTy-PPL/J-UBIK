# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2024 Max-Planck-Society

# %
from .parametric_model.parametric_prior import (
    ProbabilityConfig, transform_setting_to_prior_config)

from dataclasses import dataclass
from typing import Optional

DEFAULT_KEY = 'default'


@dataclass
class ZeroFluxPriorConfigs:
    default: ProbabilityConfig
    filters: dict[str, ProbabilityConfig]

    def get_filter_or_default(self, filter_name: str) -> ProbabilityConfig:
        '''Returns the PriorConfig for the `filter_name` or the default.

        Parameters
        ----------
        filter_name : str
            The filter in question.
        '''
        filter_name = filter_name.lower()
        if filter_name in self.filters:
            return self.filters[filter_name]
        return self.default


def yaml_to_zero_flux_prior_config(
    zero_flux_config: Optional[dict[str]]
):
    if zero_flux_config is None:
        return None

    default = transform_setting_to_prior_config(zero_flux_config[DEFAULT_KEY])

    filters = {}
    for filter_name, filter_prior in zero_flux_config.items():
        filter_name = filter_name.lower()
        if filter_name == DEFAULT_KEY:
            continue

        filters[filter_name] = transform_setting_to_prior_config(filter_prior)

    return ZeroFluxPriorConfigs(default=default, filters=filters)
