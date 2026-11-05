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


@dataclass
class FlagUV(StaticTyped):
    """Remove the rows with ``|u|`` or ``|v|`` above `max` wavelengths, see
    `flag_uv`."""

    max: float

    @classmethod
    def from_yaml_dict(cls, raw: dict | None) -> Union["FlagUV", None]:
        if not raw:
            return None
        flag = cls(max=float(raw["max"]))
        if not flag.max > 0:
            raise ValueError(f"flag_uv: max must be > 0, not {flag.max}")
        return flag
