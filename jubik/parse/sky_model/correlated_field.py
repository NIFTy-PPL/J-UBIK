from dataclasses import dataclass
from typing import Callable, Union
from ..parsing_base import FromYamlDict, StaticTyped


@dataclass
class AmplitudeTotalOffsetConfig(FromYamlDict, StaticTyped):
    """Config for a single correlated field.

    Parameters
    ----------
    offset_mean: float
    offset_std: Union[tuple, Callable]
    """

    offset_mean: float
    offset_std: Union[tuple, list, Callable]

    @classmethod
    def from_yaml_dict(cls, raw: dict) -> "AmplitudeTotalOffsetConfig":
        return cls(
            offset_mean=raw["offset_mean"],
            offset_std=raw["offset_std"],
        )


@dataclass
class CfmFluctuationsConfig(FromYamlDict, StaticTyped):
    """Config for a single correlated field.

    Parameters
    ----------
    amplitude: AmplitudeTotalOffsetConfig
    fluctuations: Union[tuple, Callable]
    loglogavgslope: Union[tuple, Callable]
    flexibility: Union[tuple, Callable] (Optional)
    asperity: Union[tuple, Callable] (Optional)
    prefix: str
    harmonic_type: str = "fourier" (Optional)
    non_parametric_kind: str = "amplitude" (Optional)
    """

    amplitude: AmplitudeTotalOffsetConfig
    fluctuations: Union[tuple, list, Callable]
    loglogavgslope: Union[tuple, list, Callable]
    flexibility: Union[tuple, list, Callable, None] = None
    asperity: Union[tuple, list, Callable, None] = None
    prefix: str = ""
    harmonic_type: str = "fourier"
    non_parametric_kind: str = "amplitude"

    @classmethod
    def from_yaml_dict(cls, raw: dict) -> "CfmFluctuationsConfig":
        flucts = raw["fluctuations"]
        return cls(
            amplitude=AmplitudeTotalOffsetConfig.from_yaml_dict(raw["amplitude"]),
            fluctuations=flucts["fluctuations"],
            loglogavgslope=flucts["loglogavgslope"],
            flexibility=flucts.get("flexibility"),
            asperity=flucts.get("asperity"),
            prefix=flucts.get("prefix", ""),
            harmonic_type=flucts.get("harmonic_type", "fourier"),
            non_parametric_kind=flucts.get("non_parametric_kind", "amplitude"),
        )


@dataclass
class MaternFluctationsConfig(FromYamlDict, StaticTyped):
    """Config for a single correlated field.

    Parameters
    ----------
    amplitude: AmplitudeTotalOffsetConfig
    scale: Union[tuple, Callable]
    cutoff: Union[tuple, Callable]
    loglogslope: Union[tuple, Callable]
    renormalize_amplitude: bool
    prefix: str = ""
    harmonic_type: str = "fourier"
    non_parametric_kind: str = "amplitude"
    """

    amplitude: AmplitudeTotalOffsetConfig
    scale: Union[tuple, list, Callable]
    cutoff: Union[tuple, list, Callable]
    loglogslope: Union[tuple, list, Callable]
    renormalize_amplitude: bool
    prefix: str = ""
    harmonic_type: str = "fourier"
    non_parametric_kind: str = "amplitude"

    @classmethod
    def from_yaml_dict(cls, raw: dict) -> "MaternFluctationsConfig":
        flucts = raw["fluctuations"]
        return cls(
            amplitude=AmplitudeTotalOffsetConfig.from_yaml_dict(raw["amplitude"]),
            scale=flucts["scale"],
            cutoff=flucts["cutoff"],
            loglogslope=flucts["loglogslope"],
            renormalize_amplitude=flucts["renormalize_amplitude"],
            prefix=flucts.get("prefix", ""),
            harmonic_type=flucts.get("harmonic_type", "fourier"),
            non_parametric_kind=flucts.get("non_parametric_kind", "amplitude"),
        )


# API ----------------------------------------------------------------------------------

FLUCTUATIONS_CONFIG_BUILDER = dict(
    matern=MaternFluctationsConfig,
    matern_kernel=MaternFluctationsConfig,
    correlated_field=CfmFluctuationsConfig,
    cfm=CfmFluctuationsConfig,
)


def single_correlated_field_config_factory(
    raw: dict,
) -> Union[CfmFluctuationsConfig, MaternFluctationsConfig]:
    """Create either a `MaternFluctationsConfig` or a `CfmFluctuationsConfig`.

    Parameters
    ----------
    raw: dict
        Raw configuration dictionary.
    """

    assert len(raw) == 1, "Building of only a single correlated field is supported."
    model_name = next(iter(raw.keys())).split("_")[0].lower()
    if model_name not in FLUCTUATIONS_CONFIG_BUILDER:
        raise NotImplementedError(
            f"Invalid perturbation model '{model_name}'"
            f", supporting {list(FLUCTUATIONS_CONFIG_BUILDER.keys())}"
        )

    config_builder = FLUCTUATIONS_CONFIG_BUILDER[model_name]

    return config_builder.from_yaml_dict(raw[model_name])
