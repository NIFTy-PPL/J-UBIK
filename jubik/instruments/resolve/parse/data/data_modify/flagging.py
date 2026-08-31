from dataclasses import dataclass
from typing import Union

from ......parse.parsing_base import StaticTyped


@dataclass
class FlagWeights(StaticTyped):
    min: float = 1e-12
    max: float = 1e12

    @classmethod
    def from_yaml_dict(cls, raw: dict | None) -> Union["FlagWeights", None]:
        return (
            cls(
                min=raw.get("min", 1e-12),
                max=raw.get("max", 1e12),
            )
            if raw
            else None
        )
