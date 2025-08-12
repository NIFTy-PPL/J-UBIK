from dataclasses import dataclass
from typing import Callable, Union
from ..parsing_base import FromYamlDict, StaticTyped


@dataclass(frozen=True)
class AmplitudeTotalOffsetConfig(FromYamlDict, StaticTyped):
    """Config for a single correlated field.

    Parameters
    ----------
    offset_mean: float
    offset_std: Union[tuple, Callable]
    """

    offset_mean: float
    offset_std: Union[tuple, Callable]

    @classmethod
    def from_yaml_dict(cls, raw: dict) -> "AmplitudeTotalOffsetConfig":
        return cls(
            offset_mean=raw["offset_mean"],
            offset_std=raw["offset_std"],
        )


@dataclass(frozen=True)
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
    fluctuations: Union[tuple, Callable]
    loglogavgslope: Union[tuple, Callable]
    flexibility: Union[tuple, Callable, None] = None
    asperity: Union[tuple, Callable, None] = None
    prefix: str = ""
    harmonic_type: str = "fourier"
    non_parametric_kind: str = "amplitude"

    @classmethod
    def from_yaml_dict(cls, raw: dict) -> "CfmFluctuationsConfig":
        return cls(
            amplitude=AmplitudeTotalOffsetConfig.from_yaml_dict(raw["amplitude"]),
            fluctuations=raw["fluctuations"],
            loglogavgslope=raw["loglogavgslope"],
            flexibility=raw.get("flexibility"),
            asperity=raw.get("asperity"),
            prefix=raw.get("prefix", ""),
            harmonic_type=raw.get("harmonic_type", "fourier"),
            non_parametric_kind=raw.get("non_parametric_kind", "amplitude"),
        )


@dataclass(frozen=True)
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
    scale: Union[tuple, Callable]
    cutoff: Union[tuple, Callable]
    loglogslope: Union[tuple, Callable]
    renormalize_amplitude: bool
    prefix: str = ""
    harmonic_type: str = "fourier"
    non_parametric_kind: str = "amplitude"

    @classmethod
    def from_yaml_dict(cls, raw: dict) -> "MaternFluctationsConfig":
        return cls(
            amplitude=AmplitudeTotalOffsetConfig.from_yaml_dict(raw["amplitude"]),
            scale=raw["scale"],
            cutoff=raw["cutoff"],
            loglogslope=raw["loglogslope"],
            renormalize_amplitude=raw["renormalize_amplitude"],
            prefix=raw.get("prefix", ""),
            harmonic_type=raw.get("harmonic_type", "fourier"),
            non_parametric_kind=raw.get("non_parametric_kind", "amplitude"),
        )


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
