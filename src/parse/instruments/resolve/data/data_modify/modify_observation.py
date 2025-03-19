from dataclasses import dataclass
from configparser import ConfigParser


@dataclass
class WeightModify:
    # scheme: str
    percentage: float


@dataclass
class ObservationModify:
    """Model class for modifying the observations

    Parameters
    ----------
    time_bins: int | None
        Average the visibilities in time to N `time_bins`
    spectral_min and `spectral_max`:
        Restrict the frequency range to lie in between this range.
    spectral_bins:
        Average the visibilities of the observation to the N spectral_bins.
    weight_modify:
        A multiplicative factor used for the
    """

    time_bins: int | None
    spectral_bins: int | None
    spectral_min: float | None
    spectral_max: float | None
    spectral_restrict_to_sky_frequencies: bool
    weight_modify: WeightModify | None
    to_double_precision: bool
    testing_percentage: float | None
    restrict_to_stokes_I: bool

    @classmethod
    def from_config_parser(cls, data_cfg: ConfigParser):
        """Create an `ObservationModify` object from a yaml file.

        Parameters
        ----------
        time bins: int | None
            How many time bins the data gets restricted to.
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
        data weight modify precentage:
            The percentage of the absolute value of the visibilities to be
            added to the sigma (weight) of the visibilities (1-5)% is adviced.
        data to_double_precision: boolian | None
            Boolian that controlls if the data is cast to double precision.
        data testing percentage: float | None
            Taking a percantage of the data for testing the model.
        restrict to stokes I: bool | None
            The data will be restricted to stokes I.
        """

        TIME_BINS_KEYS = "time bins"
        SPECTRAL_BINS_KEYS = "spectral bins"

        tb = eval(data_cfg.get(TIME_BINS_KEYS, "None"))
        sb = eval(data_cfg.get(SPECTRAL_BINS_KEYS, "None"))

        # Restrict by frequencies
        smin = eval(data_cfg.get("spectral min", "None"))
        smax = eval(data_cfg.get("spectral max", "None"))
        restrict = eval(data_cfg.get("spectral restrict_to_sky_frequencies", "False"))
        _check_spectral_min_max_consistency(smin, smax)

        percentage = data_cfg.get("data weight modify percentage")
        weight_modify = (
            None if percentage is None else WeightModify(percentage=float(percentage))
        )

        to_double_precision = eval(data_cfg.get("data to_double_precision", "True"))

        testing_percentage = eval(data_cfg.get("data testing percentage", "None"))

        restrict_to_stokes_I = eval(data_cfg.get("restrict to stokes I", "False"))

        return ObservationModify(
            time_bins=tb,
            spectral_bins=sb,
            spectral_min=smin,
            spectral_max=smax,
            spectral_restrict_to_sky_frequencies=restrict,
            weight_modify=weight_modify,
            to_double_precision=to_double_precision,
            testing_percentage=testing_percentage,
            restrict_to_stokes_I=restrict_to_stokes_I,
        )

    @classmethod
    def from_yaml_dict(cls, data_cfg: dict):
        """Create an `ObservationModify` object from a yaml file.

        Parameters
        ----------
        time_bins: int | None
            How many time bins the data gets restricted to.
        spectral: dict | None
            - bins: int | None
                If given the data gets restricted to N spectral bins.
            - min: float | None
                If given the data gets restricted to being above this minimum
                frequency value.
            - max: float | None
                If given the data gets restricted to being below this maximum
                frequency value.
            - restrict_to_sky_frequencies: bool | None
                Boolian to restrict the spectral channels of the data to the sky
                model frequencies.
        weight_modify: dict | None
            - percentage: float | None
                The percentage of the absolute value of the visibilities to be
                added to the sigma (weight) of the visibilities (1-5)% is adviced.
        to_double_precision: boolian | None
            Boolian that controlls if the data is cast to double precision.
        testing_percentage: float | None
            Taking a percantage of the data for testing the model.
        restrict_to_stokes_I: bool | None
            The data will be restricted to stokes I.
        """
        tb = data_cfg.get("time_bins")

        spectral = data_cfg.get("spectral", {})
        sb = spectral.get("bins")
        smin = spectral.get("min")
        smax = spectral.get("max")
        restrict = spectral.get("restrict_to_sky_frequencies", False)
        _check_spectral_min_max_consistency(smin, smax)

        wm = data_cfg.get("weight_modify", {})
        percentage = wm.get("percentage")
        weight_modify = (
            None if percentage is None else WeightModify(percentage=float(percentage))
        )

        to_double_precision = data_cfg.get("to_double_precision", True)
        testing_percentage = data_cfg.get("testing_percentage", None)
        restrict_to_stokes_I = data_cfg.get("restrict_to_stokes_I", False)

        return ObservationModify(
            time_bins=tb,
            spectral_bins=sb,
            spectral_min=smin,
            spectral_max=smax,
            spectral_restrict_to_sky_frequencies=restrict,
            weight_modify=weight_modify,
            to_double_precision=to_double_precision,
            testing_percentage=testing_percentage,
            restrict_to_stokes_I=restrict_to_stokes_I,
        )


def _check_spectral_min_max_consistency(smin: float, smax: float):
    if (smin is None) and (smax is None):
        return
    elif (smin is not None) and (smax is not None):
        if smin >= smax:
            raise ValueError(
                "spectral minimum must be strictly lower than " "spectral maximum."
            )
        return
    else:
        raise ValueError(
            "Both 'spectral minimum' and 'spectral maximum' needs"
            " to be set, simultanously."
        )
