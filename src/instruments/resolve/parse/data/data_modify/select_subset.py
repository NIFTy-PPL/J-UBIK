import warnings
from dataclasses import dataclass
from configparser import ConfigParser


@dataclass
class SelectSubset:
    percentage: float | None = None
    mask_path: str | None = None

    @classmethod
    def from_yaml_dict(cls, raw: dict | float | int | None) -> "SelectSubset | None":
        if raw is None:
            return None
        if isinstance(raw, (float, int)):
            warnings.warn(
                "Using a bare float for 'testing_percentage' is deprecated. "
                "Use 'select_subset: {percentage: ..., mask_path: ...}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return cls(percentage=float(raw))
        if isinstance(raw, dict):
            return cls(
                percentage=raw.get("percentage"),
                mask_path=raw.get("mask_path"),
            )
        raise ValueError(f"Cannot parse select_subset from {type(raw)}: {raw}")

    @classmethod
    def from_config_parser(
        cls, percentage: float | None, mask_path: str | None = None
    ) -> "SelectSubset | None":
        if percentage is None and mask_path is None:
            return None
        return cls(percentage=percentage, mask_path=mask_path)
