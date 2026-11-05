import numpy as np
from astropy import units as u
from nifty.re import logger

from .....grid import Grid
from ...constants import SPEEDOFLIGHT
from ...parse.data.data_modify.flagging import BeyondNyquist, FlagWeights
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
    if not np.all(ant1 < ant2):
        raise RuntimeError(
            "Flagging a baseline assumes that the data set is sorted such that "
            "`ant1 < ant2` for every row."
        )
    ind = np.logical_and(ant1 == ant1_index, ant2 == ant2_index)
    wgt = observation._weight.copy()
    wgt[:, ind] = 0.0
    if np.sum(ind) > 0:
        antenna_names = observation.auxiliary_table("ANTENNA")["STATION"]
        ant1_name = antenna_names[ant1_index]
        ant2_name = antenna_names[ant2_index]
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
    if not np.all(ant1 < ant2):
        raise RuntimeError(
            "Flagging a station assumes that the data set is sorted such that "
            "`ant1 < ant2` for every row."
        )
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

    if not mask.any():
        raise ValueError(
            "Flagging the weights would remove all rows of the observation. "
            f"Please adapt the weight bounds ({setting.min}, {setting.max})."
        )

    visibilities = obs.vis.asnumpy()[:, mask, :]
    weights = obs.weight.asnumpy()[:, mask, :]
    # `AntennaPositions.__getitem__` masks the calibration information (ant1,
    # ant2, time) alongside uvw and handles the only_imaging case.
    antenna_positions = obs.antenna_positions[mask]

    logger.info(f"Masking {(~mask).sum()} CorruptedWeights")

    return Observation(
        antenna_positions,
        visibilities,
        weights,
        obs.legacy_polarization,
        obs.freq,
        auxiliary_tables=obs._auxiliary_tables,
    )


def uv_beyond_nyquist(observation: Observation, grid: Grid) -> np.ndarray:
    """Rows of `observation` with a baseline the pixels of `grid` cannot sample.

    The NUFFT response is periodic in uv with period 1 / pixel, so a baseline
    beyond the Nyquist frequency 1 / (2 pixel) folds onto a shorter spacing
    and returns a wrong visibility without warning. A row is beyond Nyquist
    when, at the highest frequency of the observation, ``|u| > 1 / (2 dx)`` or
    ``|v| > 1 / (2 dy)`` in wavelengths, with ``dx``, ``dy`` the pixel sizes in
    radians. As in `interferometry_response`, u pairs with the x (RA) pixel
    size and v with the y (Dec) pixel size; for square pixels this does not
    matter. Rows without any weight > 0 carry no information and are not
    counted.

    Returns
    -------
    numpy.ndarray of bool, shape ``(n_rows,)``
    """
    pixel = grid.spatial.pixel_scales_xy.to_value(u.rad)
    uv = np.abs(observation.uvw[:, :2]) * np.max(observation.freq) / SPEEDOFLIGHT
    weighted = (observation.weight.asnumpy() > 0).any(axis=(0, 2))
    return weighted & (uv > 1 / (2 * pixel)).any(axis=1)


def nyquist_message(observation: Observation, grid: Grid, name: str) -> str:
    """Describe the longest baseline beyond the Nyquist frequency of `grid`,
    see `uv_beyond_nyquist`; `observation` needs at least one such row."""
    pixel = grid.spatial.pixel_scales_xy.to_value(u.rad)
    wavelength = SPEEDOFLIGHT / np.max(observation.freq)
    beyond = uv_beyond_nyquist(observation, grid)
    extent = np.abs(observation.uvw[beyond, :2]).max(axis=0)
    axis = np.argmax(extent * pixel)
    baseline = extent[axis] / wavelength
    nyquist = 1 / (2 * pixel[axis])
    needed = (1 / (2 * baseline) * u.rad).to_value(u.mas)
    return (
        f"{name}: longest |{'uv'[axis]}| baseline {extent[axis] / 1e3:.1f} km = "
        f"{baseline:.4g} wavelengths exceeds the grid's Nyquist frequency "
        f"{nyquist:.4g} wavelengths ({beyond.sum()} of {observation.nrow} rows); pixel "
        f"{(pixel[axis] * u.rad).to_value(u.mas):.4g} mas, the baseline needs "
        f"{needed:.4g} mas or less, else the NUFFT folds it onto a shorter spacing"
    )


def restrict_to_nyquist(
    observation: Observation, grid: Grid, setting: BeyondNyquist, name: str
) -> Observation:
    """Apply the `beyond_nyquist` setting of a data block to `observation`.

    Rows beyond the Nyquist frequency of `grid` (see `uv_beyond_nyquist`) raise
    a `ValueError`, or with `setting.drop` are removed and a warning is logged.
    """
    beyond = uv_beyond_nyquist(observation, grid)
    if not beyond.any():
        return observation
    if not setting.drop:
        raise ValueError(nyquist_message(observation, grid, name))

    kept = observation[~beyond]
    longest = np.linalg.norm(kept.uvw[:, :2], axis=1).max(initial=0)
    logger.warning(
        f"{name}: dropping {beyond.sum()} of {observation.nrow} rows "
        f"({beyond.mean():.2%}) beyond the grid's Nyquist frequency, the longest "
        f"kept baseline is {longest / 1e3:.2f} km"
    )
    return kept
