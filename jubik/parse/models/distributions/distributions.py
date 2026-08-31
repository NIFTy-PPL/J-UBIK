from abc import ABC, abstractmethod
from typing import Iterable, Union
from dataclasses import dataclass, astuple
from numpy.typing import ArrayLike, NDArray
import numpy as np

DISTRIBUTION_KEY = "distribution"


class ProbabilityConfig(ABC):
    distribution: str
    transformation: str | None

    @abstractmethod
    def parameters_to_shape(self, shape: tuple):
        pass


@dataclass
class DefaultPriorConfig(ProbabilityConfig):
    distribution: str
    mean: float | ArrayLike
    sigma: float | ArrayLike
    transformation: str | None = None

    def parameters_to_shape(self, shape: tuple):
        return _shape_adjust(self.mean, shape), _shape_adjust(self.sigma, shape)

    @property
    def shape(self) -> tuple[int, ...]:
        if isinstance(self.mean, float):
            return ()
        return np.shape(self.mean)


@dataclass
class UniformPriorConfig(ProbabilityConfig):
    distribution: str
    min: float | ArrayLike
    max: float | ArrayLike
    transformation: str | None = None

    def parameters_to_shape(self, shape: tuple):
        return _shape_adjust(self.min, shape), _shape_adjust(self.max, shape)

    @property
    def shape(self) -> tuple[int, ...]:
        if isinstance(self.min, float):
            return ()
        return np.shape(self.min)


@dataclass
class InverseGammaConfig(ProbabilityConfig):
    distribution: str
    a: float | ArrayLike
    scale: float | ArrayLike
    loc: float | ArrayLike
    transformation: str | None = None

    def parameters_to_shape(self, shape: tuple):
        return self.a, self.scale, self.loc

    @property
    def shape(self) -> tuple[int, ...]:
        if isinstance(self.a, float):
            return ()
        return np.shape(self.a)


@dataclass
class DeltaPriorConfig(ProbabilityConfig):
    distribution: str
    mean: float | ArrayLike
    # NOTE: The next entry is not needed however it's convinient to comply with the
    # signature of the other Configs.
    _: float | ArrayLike = 0.0
    transformation: str | None = None

    def parameters_to_shape(self, shape: tuple):
        return (_shape_adjust(self.mean, shape),)

    @property
    def shape(self) -> tuple[int, ...]:
        if isinstance(self.mean, float):
            return ()
        return np.shape(self.mean)


def prior_config_factory(
    settings: Union[dict, tuple], shape: tuple[int] | None = None
) -> Union[
    UniformPriorConfig, InverseGammaConfig, DeltaPriorConfig, DefaultPriorConfig
]:
    """Transforms the parameter distribution `settings` into a ProbabilityConfig.
    If shape is supplied, tries to put the values into the correct shape.

    Parameters
    ----------
    settings: dict | tuple
        - If dict, it needs to contain the `distribution` and the corresponding keys.
        For example for a lognormal distribuion:
        `mean`, `sigma`, Optional[`transformation`].
        - If tuple, the first entry will be handled as the `distribution`, and the
        following the corresponding keys. For example for a lognormal distribution:
        [`lognormal`, `mean`, `sigma`, `transformation` | None].
    shape: tuple[int] | None
        Optional parameter, that will try to the supplied settings into the `shape`.
        I.e. mean and sigma.
    """

    if shape is not None:
        distribution, val1, val2, transformation = astuple(
            prior_config_factory(settings)
        )
        val1 = _cast_to_shape(val1, shape)
        val2 = _cast_to_shape(val2, shape)
        return prior_config_factory((distribution, val1, val2, transformation))

    distribution = (
        settings[DISTRIBUTION_KEY] if isinstance(settings, dict) else settings[0]
    )
    distribution = distribution.lower() if type(distribution) == str else distribution

    match distribution:
        case "uniform":
            return (
                UniformPriorConfig(**settings)
                if isinstance(settings, dict)
                else UniformPriorConfig(*settings)
            )

        case "invgamma":
            return (
                InverseGammaConfig(**settings)
                if isinstance(settings, dict)
                else InverseGammaConfig(*settings)
            )

        case "delta" | None:
            return (
                DeltaPriorConfig(**settings)
                if isinstance(settings, dict)
                else DeltaPriorConfig(*settings)
            )

        case _:
            return (
                DefaultPriorConfig(**settings)
                if isinstance(settings, dict)
                else DefaultPriorConfig(*settings)
            )


# Utils --------------------------------------------------------------------------------


def _cast_to_shape(value: ArrayLike, shape: tuple[int, ...]) -> NDArray:
    """
    Return `value` with the requested `shape`:

    • broadcast where possible
    • otherwise reshape when the element count matches
    • otherwise raise ValueError.
    """
    arr = np.asarray(value)

    # already the right shape
    if arr.shape == shape:
        return arr

    # 1) try broadcasting
    try:
        return np.broadcast_to(arr, shape)
    except ValueError:
        pass  # will try reshape next

    # 2) try reshape
    if arr.size == np.prod(shape):
        # reshape never copies unless needed
        return arr.reshape(shape)

    # 3) nothing worked
    raise ValueError(
        f"cannot cast array from shape {arr.shape} to requested shape {shape}"
    )


def _shape_adjust(val, shape):
    """Adjusts the shape of the prior."""
    if np.shape(val) == shape:
        return np.array(val)
    else:
        return np.full(shape, val)
