from dataclasses import dataclass
from typing import Union

from astropy.units import temperature

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
from ..parametric_model.gaussian import GaussianConfig
from .spectral_product_mf_sky import SimpleSpectralSkyConfig


@dataclass
class BlackBodyConfig(FromYamlDict):
    """
    Parameters
    ----------
    temperature_gaussian: GaussianConfig | None
        (Optional) Gaussian mean model of the temperature field.
    temperature: Union[MaternFluctationsConfig, CfmFluctuationsConfig, DefaultPriorConfig, DeltaPriorConfig ]
        The settings of the temperature or log-temperature field.
    """

    temperature_gaussian: GaussianConfig | None
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
            - temperature_gaussian: config (optional)
        """
        # Handle optional parametric mean model for the temperature field.
        tg_setting = raw.get("temperature_gaussian")
        tg = None if tg_setting is None else GaussianConfig.from_yaml_dict(tg_setting)

        if isinstance(raw["temperature"], list):
            temperature = prior_config_factory(tuple(raw["temperature"]))
        else:
            temperature = single_correlated_field_config_factory(raw["temperature"])

        return cls(
            temperature_gaussian=tg,
            temperature=temperature,
        )


@dataclass
class ModifiedBlackBodyConfig(FromYamlDict):
    temperature_gaussian: GaussianConfig | None
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
            - temperature_gaussian: config (optional)
            - optical_depth: SimpleSpectralSkyConfig
                See the parameters of `SimpleSpectralSkyConfig`
        """
        # Optional parametric mean model for the temperature field.
        # Only Gaussian implemented.
        tg_setting = raw.get("temperature_gaussian")
        tg = None if tg_setting is None else GaussianConfig.from_yaml_dict(tg_setting)

        return cls(
            temperature_gaussian=tg,
            temperature=single_correlated_field_config_factory(raw["temperature"]),
            optical_depth=SimpleSpectralSkyConfig.from_yaml_dict(raw["optical_depth"]),
        )
