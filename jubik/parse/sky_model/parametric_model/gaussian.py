from dataclasses import dataclass

from ....instruments.jwst.parse.parametric_model.parametric_prior import (
    ProbabilityConfig,
    prior_config_factory,
)
from ...parsing_base import FromYamlDict, StaticTyped


@dataclass
class GaussianConfig(FromYamlDict, StaticTyped):
    i0: ProbabilityConfig
    center: ProbabilityConfig
    covariance: ProbabilityConfig
    off_diagonal: ProbabilityConfig

    @classmethod
    def from_yaml_dict(cls, raw: dict) -> "GaussianConfig":
        return cls(
            i0=prior_config_factory(raw["i0"], shape=()),
            center=prior_config_factory(raw["center"], shape=(2,)),
            covariance=prior_config_factory(raw["covariance"], shape=(2,)),
            off_diagonal=prior_config_factory(raw["off_diagonal"], shape=()),
        )
