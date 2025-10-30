from dataclasses import dataclass
from configparser import ConfigParser

from .shift_observation import ShiftObservation
from .frequency_handling import SpectralModify
from .flagging import FlagWeights
from .modify_weight import SystematicErrorBudget


@dataclass
class ObservationModify:
    """Model class for modifying the observations

    Parameters
    ----------
    time_bins: int | None
        Average the visibilities in time to N `time_bins`
    spectral:
        Settings for limiting the spectral range of the observation.
    spectral_bins:
        Average the visibilities of the observation to the N spectral_bins.
    weight_modify:
        Modification for the weight column (e.g. systematic error budget)
    """

    time_bins: int | None

    spectral: SpectralModify
    weight_modify: SystematicErrorBudget | None
    shift: ShiftObservation | None

    to_double_precision: bool
    testing_percentage: float | None
    restrict_to_stokes_I: bool
    average_to_stokes_I: bool
    flag_weights: FlagWeights | None

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
        systematic error budget:
            The percentage of the absolute value of the visibilities to be
            added to the sigma (weight) of the visibilities (1-5)% is adviced.
        data to_double_precision: boolian | None
            Boolian that controlls if the data is cast to double precision.
        data testing percentage: float | None
            Taking a percantage of the data for testing the model.
        restrict to stokes I: bool | None
            The data will be restricted to stokes I.
        average to stokes I: bool | None
            The data will be averaged to stokes I.
        """

        TIME_BINS_KEYS = "time bins"

        tb = eval(data_cfg.get(TIME_BINS_KEYS, "None"))

        spectral_modify = SpectralModify.from_config_parser(data_cfg)
        weight_modify = SystematicErrorBudget.from_config_parser(data_cfg)

        to_double_precision = eval(data_cfg.get("data to_double_precision", "True"))

        testing_percentage = eval(data_cfg.get("data testing percentage", "None"))

        restrict_to_stokes_I = eval(data_cfg.get("restrict to stokes I", "False"))
        average_to_stokes_I = eval(data_cfg.get("average to stokes I", "False"))

        return ObservationModify(
            time_bins=tb,
            spectral=spectral_modify,
            weight_modify=weight_modify,
            shift=None,  # NOTE: Needs ot be implemented
            to_double_precision=to_double_precision,
            testing_percentage=testing_percentage,
            restrict_to_stokes_I=restrict_to_stokes_I,
            average_to_stokes_I=average_to_stokes_I,
            flag_weights=None,  # NOTE : Needs to be implemented
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
        shift: dict | None
            - data_template: list[tuple[float, float]]
                The shift for each data_template.
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
        average_to_stokes_I: bool | None
            The data will be averaged to stokes I.
        flag_weights: dict | None
            Mask visibilities and weights according to specification
            flag_weights = dict(min=1e-12, max=1e12)
        """

        tb = data_cfg.get("time_bins")

        spectral_modify = SpectralModify.from_yaml_dict(data_cfg.get("spectral", {}))

        # TODO: Build more versions by extending modify_weight.py
        weight_modify = SystematicErrorBudget.from_yaml_dict(
            data_cfg.get("weight_modify", {})
        )
        shift = ShiftObservation.from_yaml_dict(data_cfg.get("shift"))

        to_double_precision = data_cfg.get("to_double_precision", True)
        testing_percentage = data_cfg.get("testing_percentage", None)
        restrict_to_stokes_I = data_cfg.get("restrict_to_stokes_I", False)
        average_to_stokes_I = data_cfg.get("average_to_stokes_I", False)

        flag_weights = FlagWeights.from_yaml_dict(data_cfg.get("flag_weights"))

        return ObservationModify(
            time_bins=tb,
            spectral=spectral_modify,
            weight_modify=weight_modify,
            shift=shift,
            to_double_precision=to_double_precision,
            testing_percentage=testing_percentage,
            restrict_to_stokes_I=restrict_to_stokes_I,
            average_to_stokes_I=average_to_stokes_I,
            flag_weights=flag_weights,
        )

    def __call__(self, iteration: int) -> "ObservationModify":
        return ObservationModify(
            time_bins=self.time_bins,
            spectral=self.spectral,
            weight_modify=self.weight_modify,
            shift=ShiftObservation(self.shift(iteration)) if self.shift else None,
            to_double_precision=self.to_double_precision,
            testing_percentage=self.testing_percentage,
            restrict_to_stokes_I=self.restrict_to_stokes_I,
            average_to_stokes_I=self.average_to_stokes_I,
            flag_weights=self.flag_weights,
        )
