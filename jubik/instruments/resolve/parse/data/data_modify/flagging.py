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
class BeyondNyquist(StaticTyped):
    """What to do with rows beyond the Nyquist frequency of the sky grid, see
    `restrict_to_nyquist`. With `drop` False such a row raises."""

    drop: bool = False

    @classmethod
    def from_yaml_dict(cls, raw: str | None) -> "BeyondNyquist":
        """Build from `raise` (the default for None) or `drop`."""
        policy = "raise" if raw is None else raw
        if policy not in ("raise", "drop"):
            raise ValueError(f"beyond_nyquist must be `raise` or `drop`, not {policy!r}")
        return cls(drop=policy == "drop")
