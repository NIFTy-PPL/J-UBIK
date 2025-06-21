from dataclasses import dataclass
from abc import ABC, abstractclassmethod

from .parametric_model.parametric_prior import ProbabilityConfig, prior_config_factory


# ABC for Configs ----------------------------------------------------------------------


class VariableCovarianceConfig(ABC):
    @abstractclassmethod
    def from_yaml_dict(cls, raw: dict | list):
        pass


# Concrete Implementations -------------------------------------------------------------


@dataclass
class MultiplicativeStdValueConfig(VariableCovarianceConfig):
    value: ProbabilityConfig

    @classmethod
    def from_yaml_dict(
        cls, value_settings: dict | list
    ) -> "MultiplicativeStdValueConfig":
        return cls(value=prior_config_factory(value_settings))


# API ----------------------------------------------------------------------------------


def variable_covariance_config_factory(
    raw: dict | None,
) -> MultiplicativeStdValueConfig | None:
    """Create the config class for a VariableCovariance builder.

    Parameters
    ----------
    raw: dict | None
        The configs for the variable covariance. If `None`, `None` gets returned.
    """

    if raw is None:
        return None

    mapping = {"multiplicative_std_value": MultiplicativeStdValueConfig}

    # TODO : Implement more complicated versions, where one can get more than one Covarince estimator.
    assert len(raw.keys()) == 1
    key, val = next(iter(raw.items()))

    try:
        return mapping[key].from_yaml_dict(val)
    except KeyError:
        raise KeyError(f"{key} is not implemented. Choose from {mapping.keys()}")
