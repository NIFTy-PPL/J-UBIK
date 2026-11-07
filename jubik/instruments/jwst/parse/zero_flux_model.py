# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2024 Max-Planck-Society

# %
from ....parse.models.distributions import (
    ProbabilityConfig,
    prior_config_factory,
)

from dataclasses import dataclass

DEFAULT_KEY = "default"


def match_band_key(keys, band: str) -> str | None:
    """`band` if it is one of `keys`, else the longest key ending in ``_*``
    that `band` starts with (``keck_nirc2_h_*`` for ``keck_nirc2_h_n0042``)."""
    if band in keys:
        return band
    patterns = [k for k in keys if k.endswith("_*") and band.startswith(k[:-1])]
    return max(patterns, key=len, default=None)


@dataclass
class ZeroFluxPriorConfigs:
    default: ProbabilityConfig
    names: dict[str, ProbabilityConfig]

    def get_name_setting_or_default(self, filter_name: str) -> ProbabilityConfig:
        """Returns the PriorConfig for the `filter_name`, else for the longest
        ``_*`` pattern it starts with (`match_band_key`), else the default.

        Parameters
        ----------
        filter_name : str
            The filter in question.
        """
        key = match_band_key(self.names, filter_name.lower())
        return self.default if key is None else self.names[key]

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
