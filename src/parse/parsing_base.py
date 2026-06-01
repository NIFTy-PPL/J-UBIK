from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Union, get_args, get_origin, get_type_hints


class FromYamlDict(ABC):
    @classmethod
    @abstractmethod
    def from_yaml_dict(cls, raw: dict) -> "FromYamlDict": ...


def _matches_type(value, annotation):
    if annotation is Any:
        return True

    args = get_args(annotation)

    if args:
        origin = get_origin(annotation)

        if origin is not None:
            if "Union" in str(origin):
                return any(
                    _matches_type(value, t)
                    for t in args
                )

            return isinstance(value, origin)

    return isinstance(value, annotation)


@dataclass
class StaticTyped:
    def __post_init__(self):
        self.static_typed(self)

    @staticmethod
    def static_typed(obj):
        for name, field_type in get_type_hints(type(obj)).items():
            value = getattr(obj, name)

            if not _matches_type(value, field_type):
                raise TypeError(
                    f"The field `{name}` was assigned by "
                    f"`{type(value)}` instead of `{field_type}`"
                )