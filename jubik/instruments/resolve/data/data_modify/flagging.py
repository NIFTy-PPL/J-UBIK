import numpy as np
from nifty.re import logger

from ...parse.data.data_modify.flagging import FlagWeights
from ..antenna_positions import AntennaPositions
from ..observation import Observation


def flag_baseline(
    observation: Observation, ant1_index: int, ant2_index: int
) -> Observation:
    ant1 = observation.antenna_positions.ant1
    ant2 = observation.antenna_positions.ant2
    if ant1 is None or ant2 is None:
        raise RuntimeError(
            "The calibration information needed for flagging a baseline is not "
            "available. Please import the measurement set with "
            "`with_calib_info=True`."
        )
    assert np.all(ant1 < ant2)
    ind = np.logical_and(ant1 == ant1_index, ant2 == ant2_index)
    wgt = observation._weight.copy()
    wgt[:, ind] = 0.0
    antenna_names = observation.auxiliary_table("ANTENNA")["STATION"]
    ant1_name = antenna_names[ant1_index]
    ant2_name = antenna_names[ant2_index]
    if np.sum(ind) > 0:
        print(
            f"INFO: Flag baseline {ant1_name}-{ant2_name}, {np.sum(ind)}/{observation.nrow} rows flagged."
        )
    return Observation(
        observation._antpos,
        observation._vis,
        wgt,
        observation._polarization,
        observation._freq,
        observation._auxiliary_tables,
    )


def flag_station(observation: Observation, ant_index: int) -> Observation:
    ant1 = observation.antenna_positions.ant1
    ant2 = observation.antenna_positions.ant2
    if ant1 is None or ant2 is None:
        raise RuntimeError(
            "The calibration information needed for flagging a baseline is not "
            "available. Please import the measurement set with "
            "`with_calib_info=True`."
        )
    assert np.all(ant1 < ant2)
    ind = np.logical_or(ant1 == ant_index, ant2 == ant_index)
    wgt = observation._weight.copy()
    wgt[:, ind] = 0.0
    if np.sum(ind) > 0:
        ant_name = observation.auxiliary_table("ANTENNA")["STATION"][ant_index]
        print(
            f"INFO: Flag station {ant_name}, {np.sum(ind)}/{observation.nrow} rows flagged."
        )
    return Observation(
        observation._antpos,
        observation._vis,
        wgt,
        observation._polarization,
        observation._freq,
        observation._auxiliary_tables,
    )


def flag_weights(obs: Observation, setting: FlagWeights | None) -> Observation:
    """Mask visibilities and weights according to setting.min, setting.max."""

    if setting is None:
        return obs

    mask_bigger = obs.weight.asnumpy() > setting.max
    mask_smaller = obs.weight.asnumpy() < setting.min
    mask = ~(mask_bigger + mask_smaller).any(axis=(0, 2))

    visibilities = obs.vis.asnumpy()[:, mask, :]
    weights = obs.weight.asnumpy()[:, mask, :]
    uvw = obs.antenna_positions.uvw[mask, :]

    logger.info(f"Masking {(~mask).sum()} CorruptedWeights")

    return Observation(
        AntennaPositions(
            uvw,
            obs.antenna_positions.ant1,
            obs.antenna_positions.ant2,
            obs.antenna_positions.time,
        ),
        visibilities,
        weights,
        obs.legacy_polarization,
        obs.freq,
        auxiliary_tables=obs._auxiliary_tables,
    )
