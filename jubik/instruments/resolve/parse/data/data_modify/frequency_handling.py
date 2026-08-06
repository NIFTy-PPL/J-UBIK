from configparser import ConfigParser
from dataclasses import dataclass


@dataclass
class SpectralModify:
    spectral_bins: int | None = None
    spectral_min: float | None = None
    spectral_max: float | None = None
    spectral_restrict_to_sky_frequencies: bool = False
    exclude_ranges: list | None = None

    @classmethod
    def from_config_parser(cls, data_cfg: ConfigParser) -> "SpectralModify":
        """
        Build from ConfigParser.

        Parameters
        ----------
        spectral bins: int | None
            If given the data gets restricted to N spectral bins.
        spectral min: float | None
            If given the data gets restricted to being above this minimum
            frequency value.
        spectral max: float | None
            If given the data gets restricted to being below this maximum
            frequency value.
        spectral restrict_to_sky_frequencies: bool | None
            Boolian to restrict the spectral channels of the data to the sky
            model frequencies.
        """

        sb = eval(data_cfg.get("spectral bins", "None"))
        smin = eval(data_cfg.get("spectral min", "None"))
        smax = eval(data_cfg.get("spectral max", "None"))
        restrict = eval(data_cfg.get("spectral restrict_to_sky_frequencies", "False"))
        cls._check_spectral_min_max_consistency(smin, smax)

        return cls(
            spectral_bins=sb,
            spectral_min=smin,
            spectral_max=smax,
            spectral_restrict_to_sky_frequencies=restrict,
        )

    @classmethod
    def from_yaml_dict(cls, spectral: dict) -> "SpectralModify":
        """
        Build from ConfigParser.

        Parameters
        ----------
        spectral: dict
            - bins: int | None
                If given the spectral dimension of the data gets averaged to N spectral
                bins.
            - min: float | None
                If given the data gets restricted to being above this minimum
                frequency value.
            - max: float | None
                If given the data gets restricted to being below this maximum
                frequency value.
            - restrict_to_sky_frequencies: bool | None
                Boolian to restrict the spectral channels of the data to the sky
                model frequencies.
            - exclude_frequency_ranges: list | None
                If given, a list of [fmin, fmax] pairs, in [Hz]. Every data
                channel which falls inside one of these ranges gets dropped.
                The bounds are inclusive.
        """
        sb = spectral.get("bins")
        smin = spectral.get("min")
        smax = spectral.get("max")
        restrict = spectral.get("restrict_to_sky_frequencies", False)
        exclude = spectral.get("exclude_frequency_ranges", None)
        cls._check_spectral_min_max_consistency(smin, smax)
        cls._check_exclude_ranges(exclude)

        return cls(
            spectral_bins=sb,
            spectral_min=smin,
            spectral_max=smax,
            spectral_restrict_to_sky_frequencies=restrict,
            exclude_ranges=exclude,
        )

    @staticmethod
    def _check_exclude_ranges(exclude: list | None):
        if exclude is None:
            return

        for entry in exclude:
            if not hasattr(entry, "__len__") or len(entry) != 2:
                raise ValueError(
                    "Every entry of 'exclude_frequency_ranges' must be a pair"
                    f" of [fmin, fmax], got {entry}."
                )
            fmin, fmax = entry
            if fmin >= fmax:
                raise ValueError(
                    "In every entry of 'exclude_frequency_ranges' fmin must be"
                    f" strictly lower than fmax, got {entry}."
                )

    @staticmethod
    def _check_spectral_min_max_consistency(smin: float | None, smax: float | None):
        if (smin is None) and (smax is None):
            return
        elif (smin is not None) and (smax is not None):
            if smin >= smax:
                raise ValueError(
                    "spectral minimum must be strictly lower than spectral maximum."
                )
            return
        else:
            raise ValueError(
                "Both 'spectral minimum' and 'spectral maximum' needs"
                " to be set, simultanously."
            )
