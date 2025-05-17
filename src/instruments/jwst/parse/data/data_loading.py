from collections import UserDict
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Literal


class DataFilePaths(UserDict[str, tuple[Path]]):
    """Dict-like mapping {filter → list[Path]}."""

    @classmethod
    def from_yaml_dict(cls, yml: dict) -> "DataFilePaths":
        step_type = yml.get("step_type", "cal")
        filter_spec = yml.get("filter", {})

        mapping = {
            flt: tuple(Path(p.format(step_type=step_type)) for p in raw_paths)
            for flt, raw_paths in filter_spec.items()
        }
        return cls(mapping)

    def filters(self) -> list[str]:
        return list(self.keys())


class LoadingMode(Enum):
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


@dataclass
class DataLoadingConfig:
    paths: DataFilePaths
    loading_mode: LoadingMode = LoadingMode.SERIAL
    workers: int | None = None

    @classmethod
    def from_yaml_dict(cls, yml: dict) -> "DataLoadingConfig":
        loading_mode_str: str = yml.get("loading_mode", "serial")
        workers: int | None = yml.get("workers")
        return cls(
            paths=DataFilePaths.from_yaml_dict(yml),
            loading_mode=LoadingMode.from_string(loading_mode_str),
            workers=workers,
        )
