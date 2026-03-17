import warnings
from dataclasses import dataclass
from configparser import ConfigParser


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
        """Create a `SelectSubset` from ConfigParser values.

        Parameters
        ----------
        percentage: float | None
            Fraction of visibilities to keep.
        mask_path: str | None
            Path to a .npy file for saving/loading the subset mask.

        Returns
        -------
        SelectSubset | None
            None if both arguments are None, otherwise a `SelectSubset` instance.
        """
        if percentage is None and mask_path is None:
            return None
        return cls(percentage=percentage, mask_path=mask_path)
