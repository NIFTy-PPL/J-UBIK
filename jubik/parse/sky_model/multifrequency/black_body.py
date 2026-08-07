from dataclasses import dataclass
from typing import Union

from ....instruments.jwst.parse.parametric_model.parametric_prior import (
    DefaultPriorConfig,
    DeltaPriorConfig,
    prior_config_factory,
)
from ...parsing_base import FromYamlDict
from ..correlated_field import (
    CfmFluctuationsConfig,
    MaternFluctationsConfig,
    single_correlated_field_config_factory,
)
from .spectral_product_mf_sky import SimpleSpectralSkyConfig


@dataclass
class BlackBodyConfig(FromYamlDict):
    """
    Parameters
    ----------
    temperature: Union[MaternFluctationsConfig, CfmFluctuationsConfig, DefaultPriorConfig, DeltaPriorConfig ]
        The settings of the temperature or log-temperature field.
    """

    temperature: Union[
        MaternFluctationsConfig,
        CfmFluctuationsConfig,
        DefaultPriorConfig,
        DeltaPriorConfig,
    ]

    @property
    def is_field(self):
        """Specifying if this corresponds to config for a field or a single value."""
        if self.temperature in [MaternFluctationsConfig, CfmFluctuationsConfig]:
            return True
        else:
            return False

    @classmethod
    def from_yaml_dict(cls, raw: dict) -> "BlackBodyConfig":
        """Initialization from the raw dictionary.

        Parameters
        ----------
        raw: dict
            - temperature: config
                Single valued : ["lognormal", "invgamma", "delta"]
                Field valued : ["matern", "cfm"]
        """
        if isinstance(raw["temperature"], list):
            temperature = prior_config_factory(tuple(raw["temperature"]))
        else:
            temperature = single_correlated_field_config_factory(raw["temperature"])

        return cls(
            temperature=temperature,
        )


@dataclass
class ModifiedBlackBodyConfig(FromYamlDict):
    temperature: MaternFluctationsConfig | CfmFluctuationsConfig
    optical_depth: SimpleSpectralSkyConfig

    @classmethod
    def from_yaml_dict(cls, raw: dict) -> "ModifiedBlackBodyConfig":
        """Initialization from the raw dictionary.

        Parameters
        ----------
        raw: dict
            - temperature: config
                Field valued : ["matern", "cfm"]
            - optical_depth: SimpleSpectralSkyConfig
                See the parameters of `SimpleSpectralSkyConfig`
        """
        return cls(
            temperature=single_correlated_field_config_factory(raw["temperature"]),
            optical_depth=SimpleSpectralSkyConfig.from_yaml_dict(raw["optical_depth"]),
        )
