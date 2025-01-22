from abc import ABC
from typing import Union, Optional
from dataclasses import dataclass

DISTRIBUTION_KEY = 'distribution'


class ProbabilityConfig(ABC):
    distribution: str
    transformation: Optional[str]

    def to_list(self) -> list:
        '''Return the ProbabilityConfig as a tuple.'''
        # NOTE: __annotations__ keeps the order of the fields of the
        # make_dataclass used inside create_distribution_config_class
        return list(getattr(self, field) for field in self.__annotations__.keys())


@dataclass
class DefaultPriorConfig(ProbabilityConfig):
    distribution: str
    mean: float
    sigma: float
    transformation: Optional[str] = None


@dataclass
class UniformPriorConfig(ProbabilityConfig):
    distribution: str
    min: float
    max: float
    transformation: Optional[str] = None


@dataclass
class DeltaPriorConfig(ProbabilityConfig):
    distribution: str
    mean: float
    _: Optional[float] = None  # This is not needed however it's convinient to
    # comply with the signature of the other Configs.
    transformation: Optional[str] = None


def transform_setting_to_prior_config(settings: Union[dict, tuple]):
    """Transforms the parameter distribution `settings` into a ProbabilityConfig."""

    distribution = (settings[DISTRIBUTION_KEY] if isinstance(settings, dict)
                    else settings[0])
    distribution = (distribution.lower() if type(distribution) == str
                    else distribution)

    match distribution:
        case 'uniform':
            return (
                UniformPriorConfig(**settings) if isinstance(settings, dict)
                else UniformPriorConfig(*settings)
            )

        case 'delta' | None:
            return (
                DeltaPriorConfig(**settings) if isinstance(settings, dict)
                else DeltaPriorConfig(*settings)
            )

        case _:
            return (
                DefaultPriorConfig(**settings) if isinstance(settings, dict)
                else DefaultPriorConfig(*settings)
            )
