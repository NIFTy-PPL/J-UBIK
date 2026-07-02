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

    @classmethod
    def from_yaml_dict(cls, raw: dict | None):
        """Read the ZerofluxPriorConfigs from parsed yaml file

        Parameters
        ----------
        raw: dict, Parsed dict from yaml file, containing:
            - default, the default prior settings
            - names, optional, other names corresponding to different filters where a
                     zeroflux-model will be applied.
        """

        if raw is None:
            return None

        default = prior_config_factory(raw[DEFAULT_KEY])

        names = {}
        for filter_name, filter_prior in raw.items():
            filter_name = filter_name.lower()
            if filter_name == DEFAULT_KEY:
                continue

            names[filter_name] = prior_config_factory(filter_prior)

        return ZeroFluxPriorConfigs(default=default, names=names)
