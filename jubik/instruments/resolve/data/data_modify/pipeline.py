"""Pipeline for applying configured transformations to an observation."""

from dataclasses import dataclass

import logging
import astropy.units as u
from nifty.re import logger

from .....color import Color
from ...parse.data.data_loading import DataFile
from ...parse.data.data_modify import ObservationModifyConfig
from ...parse.data.data_modify.flagging import FlagWeights
from ...parse.data.data_modify.frequency import SpectralModify
from ...parse.data.data_modify.visibility_subset import SelectSubset
from ...parse.data.data_modify.weights import SystematicErrorBudget
from ...conventions import OBSERVATION_FREQ_UNIT
from ..observation import Observation
from .flagging import flag_weights
from .frequency import (
    exclude_frequency_ranges,
    freq_average_by_fdom_and_n_freq_chunks,
    restrict_by_freq,
    restrict_to_discontinuous_frequencies,
    reverse_frequencies,
)
from .polarization import average_stokesi, restrict_to_stokesi
from .precision import to_double_precision, to_single_precision
from .visibility_subset import select_random_visibility_subset
from .time import time_average_to_length_of_timebins
from .weights import systematic_error_budget
from .phase_center import ShiftObservation, shift_phase_center


@dataclass
class ObservationModify:
    """The modification of one observation, see `modify_observation`.

    Parameters
    ----------
    time_bins: int | None
        Average the visibilities in time to this many bins.
    spectral: SpectralModify
        Restrict, exclude and average the frequency channels.
    weight_modify: SystematicErrorBudget | None
        Systematic error budget added to the noise of the visibilities.
    shift: ShiftObservation | None
        Fixed phase-center shift of the observation.
    to_double_precision: bool
        Cast visibilities and weights to double precision.
    select_subset: SelectSubset | None
        Keep a random subset of the visibilities.
    restrict_to_stokes_I: bool
        Restrict the data to Stokes I.
    average_to_stokes_I: bool
        Average the data to Stokes I.
    flag_weights: FlagWeights | None
        Flag visibilities whose weights lie outside a range.
    """

    time_bins: int | None

    spectral: SpectralModify
    weight_modify: SystematicErrorBudget | None
    shift: ShiftObservation | None

    to_double_precision: bool
    select_subset: SelectSubset | None
    restrict_to_stokes_I: bool
    average_to_stokes_I: bool
    flag_weights: FlagWeights | None


def build_observation_modify(
    config: ObservationModifyConfig, file: DataFile
) -> ObservationModify:
    """The modification of one data file of a data block.

    Parameters
    ----------
    config: ObservationModifyConfig
        The modification of the data block.
    file: DataFile
        A file of the block, from `DataLoading.data_files`.

    Returns
    -------
    ObservationModify
        The settings of `config`, with the shift of the file's data template.
    """
    return ObservationModify(
        time_bins=config.time_bins,
        spectral=config.spectral,
        weight_modify=config.weight_modify,
        shift=(
            None
            if config.shift is None
            else ShiftObservation(config.shift.shifts[file.template_index])
        ),
        to_double_precision=config.to_double_precision,
        select_subset=config.select_subset,
        restrict_to_stokes_I=config.restrict_to_stokes_I,
        average_to_stokes_I=config.average_to_stokes_I,
        flag_weights=config.flag_weights,
    )


logger.setLevel(logging.INFO)


def modify_observation(
    sky_frequencies: Color,
    obs: Observation,
    modify: ObservationModify,
) -> Observation:
    """Returns an observation according to ObservationModify. Furthermore,
    if the frequencies are not ordered from smallest to biggest the frequencies
    get reversed. Additionally the visibilities get converted to double
    precission.

    Parameters
    ----------
    sky_frequencies: Color
        The frequencies of the sky model. They are converted to the unit in
        which the observation frequencies are stored (Hz).
    obs: Observation
        The observation to be modified
    modify: ObservationModify
        The model for the modification, see `ObservationModify`.
    """

    # Masking
    obs = flag_weights(obs, modify.flag_weights)

    # Reverse the frequencies if they are ordered from high to low.
    if len(obs.freq) > 1:
        if obs.freq[1] - obs.freq[0] < 0:
            obs = reverse_frequencies(obs)

    obs = select_random_visibility_subset(obs, modify.select_subset)

    obs = time_average_to_length_of_timebins(obs, modify.time_bins)

    # TODO: Make the two cases into one and supply a None to the function!
    if modify.spectral.spectral_min is not None:
        obs = restrict_by_freq(
            obs, modify.spectral.spectral_min, modify.spectral.spectral_max
        )
    if modify.spectral.spectral_restrict_to_sky_frequencies:
        freqs_in_unit = sky_frequencies.to(
            OBSERVATION_FREQ_UNIT, equivalencies=u.spectral()
        ).value
        obs = restrict_by_freq(obs, freqs_in_unit[0], freqs_in_unit[-1])

    if not sky_frequencies.is_continuous:
        logger.info("Restrict to discontinuous Sky frequencies.")
        obs = restrict_to_discontinuous_frequencies(obs, sky_frequencies)

    # Exclusion has to happen before the frequency averaging, such that the
    # excluded channels do not enter any average.
    obs = exclude_frequency_ranges(obs, modify.spectral.exclude_ranges)

    obs = freq_average_by_fdom_and_n_freq_chunks(
        sky_frequencies,
        obs,
        modify.spectral.spectral_bins,
        averaging_mode=modify.spectral.averaging_mode,
    )
    obs = systematic_error_budget(obs, modify.weight_modify)

    # TODO: None-fy
    if modify.restrict_to_stokes_I:
        logger.info("Restrict to Stokes I")
        obs = restrict_to_stokesi(obs)

    # TODO: None-fy
    if modify.average_to_stokes_I:
        logger.info("Average to Stokes I")
        obs = average_stokesi(obs)

    # TODO: None-fy
    if modify.to_double_precision:
        obs = to_double_precision(obs)

    obs = shift_phase_center(obs, modify.shift)

    return obs
