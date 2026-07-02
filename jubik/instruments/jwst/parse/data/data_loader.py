from collections import UserDict, namedtuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Subsample(int):
    @classmethod
    def from_yaml_dict(cls, raw: dict):
        return cls(raw["subsample"])


IndexAndPath = namedtuple("IndexAndPath", ["index", "path"])


class FilterFilePaths(UserDict[str, tuple[IndexAndPath]]):
    """Dict-like mapping {filter → list[Path]}."""

    @classmethod
    def from_yaml_dict(cls, yml: dict) -> "FilterFilePaths":
        step_type = yml.get("step_type", "cal")
        filters_and_paths = yml.get("filter", {})

        mapping = {
            flt: tuple(
                IndexAndPath(index, Path(path.format(step_type=step_type)))
                for index, path in enumerate(raw_paths)
            )
            for flt, raw_paths in filters_and_paths.items()
        }
        return cls(mapping)

    def filters(self) -> list[str]:
        return list(self.keys())


class LoadingMode(Enum):
    """Enum for the different loading modes."""

    SERIAL = "serial"
    THREADS = "threads"
    PROCESSES = "processes"

    @classmethod
    def from_string(cls, mode_str: str) -> "LoadingMode":
        """Convert a string to the corresponding LoadingMode enum value."""
        try:
            return cls(mode_str.lower())
        except ValueError:
            raise ValueError(
                f"Unknown loading mode: '{mode_str}'. "
                f"Valid options are: {', '.join(m.value for m in cls)}"
            )


@dataclass(slots=True, frozen=True)
class LoadingModeConfig:
    """How the data will be loaded, with meta information.

    Parameters
    ----------
    loading_mode: LoadingMode, algorithm for loading data:
        - "serial": Sequential processing
        - "threads": Multi-threaded for I/O-bound operations
        - "processes": Multi-process for CPU-bound operations
    workers: int | None
        Number of threads/processes to use (None = executor default)
    """

    loading_mode: LoadingMode = LoadingMode.SERIAL
    workers: int | None = None

    @classmethod
    def from_yaml_dict(cls, yml: dict) -> "LoadingModeConfig":
        loading_mode_str: str = yml.get("loading_mode", "serial")
        workers: int | None = yml.get("workers")
        return cls(
            loading_mode=LoadingMode.from_string(loading_mode_str),
            workers=workers,
        )


@dataclass
class DataLoadingConfig:
    paths: FilterFilePaths
    loading_mode_config: LoadingModeConfig

    @classmethod
    def from_yaml_dict(cls, yml: dict) -> "DataLoadingConfig":
        return cls(
            paths=FilterFilePaths.from_yaml_dict(yml),
            loading_mode_config=LoadingModeConfig.from_yaml_dict(yml),
        )
