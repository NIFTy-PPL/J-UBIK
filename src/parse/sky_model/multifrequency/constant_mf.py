from dataclasses import dataclass

from ...parsing_base import FromYamlDict


@dataclass
class ConstantMFConfig(FromYamlDict):
    value: dict | tuple

    @classmethod
    def from_yaml_dict(cls, raw: dict) -> "ConstantMFConfig":
        return cls(value=raw["value"])
