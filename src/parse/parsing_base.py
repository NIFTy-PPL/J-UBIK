from dataclasses import dataclass
from abc import ABC, abstractmethod


class FromYamlDict(ABC):
    @classmethod
    @abstractmethod
    def from_yaml_dict(cls, raw: dict) -> "FromYamlDict": ...


# @dataclass(frozen=True)
class StaticTyped:
    def __post_init__(self):
        """Validate that all fields match their annotated types."""
        self.static_typed(self)

    @staticmethod
    def static_typed(obj):
        """Validate that all fields match their annotated types."""
        for name, field_type in obj.__annotations__.items():
            if not isinstance(obj.__dict__[name], field_type):
                current_type = type(obj.__dict__[name])
                raise TypeError(
                    f"The field `{name}` was assigned by "
                    f"`{current_type}` instead of `{field_type}`"
                )
