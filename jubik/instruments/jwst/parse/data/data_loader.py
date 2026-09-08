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
