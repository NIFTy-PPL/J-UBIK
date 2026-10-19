from collections import UserDict, namedtuple
from dataclasses import dataclass
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


@dataclass
class DataLoadingConfig:
    paths: FilterFilePaths

    @classmethod
    def from_yaml_dict(cls, yml: dict) -> "DataLoadingConfig":
        return cls(
            paths=FilterFilePaths.from_yaml_dict(yml),
        )


class ErrScale(UserDict[str, float]):
    """Fixed per-filter factor on the ERR arrays, `std = err * factor`.

    Filters not listed keep a factor of 1.0.
    """

    @classmethod
    def from_yaml_dict(cls, raw: dict | None, filters: list[str]) -> "ErrScale":
        if raw is None:
            return cls({})

        mapping = {}
        for flt, value in raw.items():
            if flt not in filters:
                raise ValueError(
                    f"err_scale: unknown filter '{flt}', data filters are {filters}"
                )
            value = float(value)
            if not 0 < value < float("inf"):
                raise ValueError(
                    f"err_scale: factor for '{flt}' must be positive, got {value}"
                )
            mapping[flt] = value
        return cls(mapping)
