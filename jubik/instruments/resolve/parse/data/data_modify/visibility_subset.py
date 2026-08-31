"""Configuration for visibility-subset selection."""

import warnings
from dataclasses import dataclass


@dataclass
class SelectSubset:
    """Configuration for selecting a random visibility subset.

    Parameters
    ----------
    percentage: float | None
        Fraction of visibilities to keep (e.g. 0.1 for 10%).
    mask_path: str | None
        Path to a .npy file for saving/loading the subset mask. If the
        file exists the mask is loaded from disk; otherwise a new mask is
        generated from `percentage` and saved to this path.
    """

    percentage: float | None = None
    mask_path: str | None = None

    @classmethod
    def from_yaml_dict(cls, raw: dict | float | int | None) -> "SelectSubset | None":
        """Create a `SelectSubset` from a YAML config value.

        Parameters
        ----------
        raw: dict | float | int | None
            Either a dict with keys ``percentage`` and ``mask_path``, a bare
            float/int (deprecated, treated as percentage), or None.

        Returns
        -------
        SelectSubset | None
            None if `raw` is None, otherwise a `SelectSubset` instance.
        """
        if raw is None:
            return None
        if isinstance(raw, (float, int)):
            return cls(percentage=float(raw))
        if isinstance(raw, dict):
            return cls(
                percentage=raw.get("percentage"),
                mask_path=raw.get("mask_path"),
            )
        raise ValueError(f"Cannot parse select_subset from {type(raw)}: {raw}")
