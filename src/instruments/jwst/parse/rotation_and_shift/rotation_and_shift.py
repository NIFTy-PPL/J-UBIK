from enum import Enum
from dataclasses import dataclass
from typing import Union


@dataclass
class LinearConfig:
    order: int = 1
    sky_as_brightness: bool = False
    mode: str = 'constant'


@dataclass
class NufftConfig:
    sky_as_brightness: bool = False


@dataclass
class SparseConfig:
    extend_factor: int = 1
    to_bottom_left: bool = False


class RotationAndShiftAlgorithm(Enum):
    LINEAR = LinearConfig
    NUFFT = NufftConfig
    SPARSE = SparseConfig

    @staticmethod
    def match_algorithm(
        config: Union[LinearConfig, NufftConfig, SparseConfig]
    ) -> "RotationAndShiftAlgorithm":
        """Find the matching Algorithm for the given configuration."""
        for algorithm in RotationAndShiftAlgorithm:
            if isinstance(config, algorithm.value):
                return algorithm
        raise ValueError(
            f"No matching algorithm found for config: {type(config).__name__}")


ALGORITHM_KEY = 'algorithm'
ALGORITHM_SETTINGS = 'algorithm_settings'


def yaml_to_rotation_and_shift_algorithm_config(
    rotation_and_shift_config: dict
) -> Union[LinearConfig, NufftConfig, SparseConfig]:
    """Convert a rotation_and_shift_config dictionary into a corresponding
    algorithm config object.

    Parameters
    ----------
    rotation_and_shift_config: dict
        A dictionary containing `algorithm` and `algorithm_settings`.
    """
    algorithm_name = rotation_and_shift_config.get(ALGORITHM_KEY)
    algorithm_settings = rotation_and_shift_config.get(ALGORITHM_SETTINGS, {})

    # Match the algorithm_name to the Enum
    try:
        algorithm = RotationAndShiftAlgorithm[algorithm_name.upper()]
    except KeyError:
        raise ValueError(
            f"Invalid algorithm name '{algorithm_name}'. "
            "Supported algorithms are: "
            f"{[alg.name.lower() for alg in RotationAndShiftAlgorithm]}"
        )

    # Instantiate the appropriate config class with the provided settings
    config_class = algorithm.value
    try:
        return config_class(**algorithm_settings)
    except TypeError as e:
        raise ValueError(
            f"Invalid settings for {algorithm_name}: {e}. "
            f"Expected settings: {list(config_class.__annotations__.keys())}"
        )
