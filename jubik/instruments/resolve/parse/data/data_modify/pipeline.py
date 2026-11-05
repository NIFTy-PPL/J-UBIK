"""Configuration for the observation-modification pipeline."""

from dataclasses import dataclass
import warnings

from .phase_center import ShiftObservationConfig
from .frequency import SpectralModify
from .flagging import BeyondNyquist, FlagWeights
from .weights import SystematicErrorBudget
from .visibility_subset import SelectSubset


@dataclass
class ObservationModifyConfig:
    """The modification of the observations of a data block, parsed from its
    config. `build_observation_modify` gives the modification of one file of
    the block.

    Parameters
    ----------
    shift: ShiftObservationConfig | None
        Fixed phase-center shift of each data template of the block.

    The other fields are those of `ObservationModify`, which takes them over
    unchanged.
    """

    time_bins: int | None

    spectral: SpectralModify
    weight_modify: SystematicErrorBudget | None
    shift: ShiftObservationConfig | None

    to_double_precision: bool
    select_subset: SelectSubset | None
    restrict_to_stokes_I: bool
    average_to_stokes_I: bool
    flag_weights: FlagWeights | None
    beyond_nyquist: BeyondNyquist

    @classmethod
    def from_yaml_dict(cls, data_cfg: dict) -> "ObservationModifyConfig":
        """Build from the yaml dictionary of a data block.

        Parameters
        ----------
        data_cfg: dict
            The data block, of which these keys are read:

            - time_bins: int | None
                Average the visibilities in time to this many bins.
            - spectral: dict | None
                See `SpectralModify.from_yaml_dict`.
            - weight_modify: dict | None
                - percentage: float | None
                    Percentage of the absolute value of the visibilities added
                    to their noise std, 1-5 % is advised.
            - shift: dict | None
                See `ShiftObservationConfig.from_yaml_dict`; needs the
                `data_templates` of the block.
            - to_double_precision: bool
                Cast the data to double precision, default True.
            - select_subset: dict | float | None
                Either a dict with `percentage` and `mask_path`, or a bare
                float. With `mask_path` the mask is saved to or loaded from
                that path. Replaces the deprecated `testing_percentage`.
            - restrict_to_stokes_I: bool
                Restrict the data to Stokes I, default False.
            - average_to_stokes_I: bool
                Average the data to Stokes I, default False.
            - flag_weights: dict | None
                Flag visibilities whose weights lie outside
                `[min, max]`, e.g. `dict(min=1e-12, max=1e12)`.
            - beyond_nyquist: str | None
                `raise` (default) or `drop`, see `BeyondNyquist`.
        """

        tb = data_cfg.get("time_bins")

        spectral_modify = SpectralModify.from_yaml_dict(data_cfg.get("spectral", {}))

        # TODO: Build more versions by extending weights.py
        weight_modify = SystematicErrorBudget.from_yaml_dict(
            data_cfg.get("weight_modify", {})
        )
        shift = ShiftObservationConfig.from_yaml_dict(
            data_cfg.get("shift"), len(data_cfg.get("data_templates", []))
        )

        to_double_precision = data_cfg.get("to_double_precision", True)
        if select_subset := data_cfg.get("testing_percentage"):
            warnings.warn(
                "Using a bare float for 'testing_percentage' is deprecated. "
                "Use 'select_subset: {percentage: ..., mask_path: ...}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        select_subset = SelectSubset.from_yaml_dict(
            data_cfg.get("select_subset", select_subset)
        )
        restrict_to_stokes_I = data_cfg.get("restrict_to_stokes_I", False)
        average_to_stokes_I = data_cfg.get("average_to_stokes_I", False)

        flag_weights = FlagWeights.from_yaml_dict(data_cfg.get("flag_weights"))
        beyond_nyquist = BeyondNyquist.from_yaml_dict(data_cfg.get("beyond_nyquist"))

        return ObservationModifyConfig(
            time_bins=tb,
            spectral=spectral_modify,
            weight_modify=weight_modify,
            shift=shift,
            to_double_precision=to_double_precision,
            select_subset=select_subset,
            restrict_to_stokes_I=restrict_to_stokes_I,
            average_to_stokes_I=average_to_stokes_I,
            flag_weights=flag_weights,
            beyond_nyquist=beyond_nyquist,
        )

