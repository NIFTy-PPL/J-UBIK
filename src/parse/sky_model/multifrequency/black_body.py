from dataclasses import dataclass
from typing import Union

from astropy.units import temperature

from ....instruments.jwst.parse.parametric_model.parametric_prior import (
    DefaultPriorConfig,
    DeltaPriorConfig,
    PriorConfigFactory,
)
from ...parsing_base import FromYamlDict
from ..correlated_field import (
    FLUCTUATIONS_CONFIG_BUILDER,
    CfmFluctuationsConfig,
    MaternFluctationsConfig,
    single_correlated_field_config_factory,
)
from ..parametric_model.gaussian import GaussianConfig
from .spectral_product_mf_sky import SimpleSpectralSkyConfig


@dataclass
class BlackBodyConfig(FromYamlDict):
    temperature_gaussian: GaussianConfig | None
    temperature: Union[
        MaternFluctationsConfig,
        CfmFluctuationsConfig,
        DefaultPriorConfig,
        DeltaPriorConfig,
    ]

    @property
    def is_field(self):
        if self.temperature in [MaternFluctationsConfig, CfmFluctuationsConfig]:
            return True
        else:
            return False

    @staticmethod
    def _handle_single_config(raw: dict, model_builders: dict):
        """Handle the construction of a single config builder.

        Parameters
        ----------
        raw: dict
            The config dictionary, holding ONE
             - key (model name)
             - value (model config)
        model_builders: dict
            All available models for this process with
             - key (model builder name)
             - value (model builder)
        """
        for ii, model_name_raw in enumerate(list(raw)):
            model_name = model_name_raw.split("_")[0].lower()

            # Catch not implemented errors
            if ii != 0:
                raise NotImplementedError("Only one temperature is implemented.")
            elif model_name not in model_builders:
                raise NotImplementedError(
                    f"Invalid perturbation model '{model_name}'"
                    f", supporting {list(model_builders)}"
                )

        return model_builders[model_name].from_yaml_dict(raw[model_name_raw])

    @classmethod
    def from_yaml_dict(cls, raw: dict) -> "BlackBodyConfig":
        # Handle optional parametric mean model for the temperature field.
        tg_setting = raw.get("temperature_gaussian")
        tg = None if tg_setting is None else GaussianConfig.from_yaml_dict(tg_setting)

        temperature = cls._handle_single_config(
            raw["temperature"],
            FLUCTUATIONS_CONFIG_BUILDER
            | {key: PriorConfigFactory for key in ["lognormal", "invgamma", "delta"]},
        )

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
        # Optional parametric mean model for the temperature field.
        # Only Gaussian implemented.
        tg_setting = raw.get("temperature_gaussian")
        tg = None if tg_setting is None else GaussianConfig.from_yaml_dict(tg_setting)

        return cls(
            temperature_gaussian=tg,
            temperature=single_correlated_field_config_factory(raw["temperature"]),
            optical_depth=SimpleSpectralSkyConfig.from_yaml_dict(raw["optical_depth"]),
        )
