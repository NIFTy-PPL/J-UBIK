# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2019-2021 Max-Planck-Society

import os
from os.path import expanduser, isdir, join

import numpy as np

from ..util import my_assert, my_assert_isinstance, my_asserteq
from ..logger import logger
from .antenna_positions import AntennaPositions
from .auxiliary_table import AuxiliaryTable
from .observation import Observation
from .polarization import Polarization


def ms_table(path):
    try:
        from casacore.tables import table
    except ImportError:
        raise ImportError("You need to install python-casacore for working with measurement sets")
    return table(path, readonly=True, ack=False)


def _pol_id(ms_path, spectral_window):
    """Return id for indexing polarization table for a given spectral window."""
    with ms_table(join(ms_path, "DATA_DESCRIPTION")) as t:
        polid = t.getcol("POLARIZATION_ID")[spectral_window]
    return polid


def ms2observations_all(ms, data_column):
    for spw in range(ms_n_spectral_windows(ms)):
        obs = ms2observations(ms, data_column, True, spw)
        for oo in obs:
            if oo is not None:
                yield oo


def ms2observations(ms, data_column, with_calib_info, spectral_window,
                    polarizations="all", channels=slice(None), ignore_flags=False,
                    field=None):
    """Read and convert a given measurement set into an array of :class:`Observation`

    If WEIGHT_SPECTRUM is available this column is used for weighting.
    Otherwise fall back to WEIGHT.

    Parameters
    ----------
    ms : string
        Folder name of measurement set
    data_column : string
        Column of measurement set from which the visibilities are read.
        Typically either "DATA" or "CORRECTED_DATA".
    with_calib_info : bool
        Reads in all information necessary for calibration, if True. If only
        imaging shall be performed, this can be set to False.
    spectral_window : int
        Index of spectral window that shall be imported.
    polarizations
        "all":     All polarizations are imported.
        "stokesi": Only LL/RR or XX/YY polarizations are imported.
        "stokesiavg": Only LL/RR or XX/YY polarizations are imported and averaged on the fly.
        List of strings: Strings can be "XX", "YY", "XY", "YX", "LL", "LR", "RL", "RR". The respective polarizations are loaded.
    channels : slice or list
        Select channels. Can be either a slice object or an index list. Default:
        select all channels
    ignore_flags : bool
        If True, the whole measurement set is imported irrespective of the
        flags. Default is false.
    field : str or None
        Field that is imported. If None, all fields are imported. Default: None.

    Returns
    -------
    array[Observation]
        an array of :class:`Observation` found in the measurement set

    Note
    ----
    Multiple spectral windows are not imported into one Observation instance
    because it is not guaranteed by the measurement set data structure that all
    baselines are present in all spectral windows.
    """
    # Input checks
    ms = expanduser(ms)
    if ms[-1] == "/":
        ms = ms[:-1]
    if not isdir(ms):
        raise RuntimeError
    if ms == ".":
        ms = os.getcwd()
    if isinstance(polarizations, list):
        for ll in polarizations:
            my_assert(ll in ["XX", "YY", "XY", "YX", "LL", "LR", "RL", "RR"])
        my_asserteq(len(set(polarizations)), len(polarizations))
    else:
        my_assert(polarizations in ["stokesi", "stokesiavg", "all"])
    my_assert_isinstance(channels, (slice, list))
    my_assert_isinstance(spectral_window, int)
    my_assert(spectral_window >= 0)
    my_assert(spectral_window < ms_n_spectral_windows(ms))
    # /Input checks

    # Polarization
    with ms_table(join(ms, "POLARIZATION")) as t:
        pol = t.getcol("CORR_TYPE", startrow=_pol_id(ms, spectral_window), nrow=1)[0]
        polobj = Polarization(pol)
        if polarizations == "stokesiavg":
            pol_ind = polobj.stokes_i_indices()
            polobj = Polarization.trivial()
            pol_summation = True
        elif polarizations == "all":
            pol_ind = None
            pol_summation = False
        else:
            if polarizations == "stokesi":
                polarizations = ["LL", "RR"] if polobj.circular() else ["XX", "YY"]
            pol_ind = [polobj.to_str_list().index(ii) for ii in polarizations]
            polobj = polobj.restrict_by_name(polarizations)
            pol_summation = False

    observations = []
    for ifield in range(ms_n_fields(ms)):
        auxtables = {}
        auxtables["ANTENNA"] = _import_aux_table(ms, "ANTENNA")
        auxtables["STATE"] = _import_aux_table(ms, "STATE")
        auxtables["SPECTRAL_WINDOW"] = _import_aux_table(ms, "SPECTRAL_WINDOW", row=spectral_window, skip=["ASSOC_NATURE", "ASSOC_SPW_ID"])
        sf = _source_and_field_table(ms, spectral_window, ifield)
        if sf is None:
            print(f"Field {ifield} cannot be found in SOURCE table")
            observations.append(None)
            continue
        auxtables = {**auxtables, **sf}
        source_name = auxtables['SOURCE']['NAME'][0]
        if field is not None and source_name != field:
            continue
        print(f"Work on Field {ifield}: {source_name}")
        mm = read_ms_i(ms, data_column, ifield, spectral_window, pol_ind, pol_summation,
                       with_calib_info, channels, ignore_flags)
        if mm is None:
            print(f"{ms}, field #{ifield} is empty or fully flagged")
            observations.append(None)
            continue
        # POINTING table is ignored for now
        antpos = AntennaPositions(mm["uvw"], mm["ant1"], mm["ant2"], mm["time"])
        obs = Observation(antpos, mm["vis"], mm["wgt"], polobj, mm["freq"],
                          auxiliary_tables=auxtables)
        observations.append(obs)
    return observations


def _ms2resolve_transpose(arr):
    my_asserteq(arr.ndim, 3)
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))


def _determine_weighting(ms):
    with ms_table(ms) as t:
        if "WEIGHT_SPECTRUM" in t.colnames():
            weightcol = "WEIGHT_SPECTRUM"
            fullwgt = True

            # TODO Find a better way to detect whether weight spectrum is available
            try:
                t.getcol("WEIGHT_SPECTRUM", nrow=1)
            except RuntimeError:
                weightcol = "WEIGHT"
                fullwgt = False
        else:
            weightcol = "WEIGHT"
            fullwgt = False
    return fullwgt, weightcol


def read_ms_i(name, data_column, field, spectral_window, pol_indices, pol_summation,
              with_calib_info, channels, ignore_flags):
    assert pol_indices is None or isinstance(pol_indices, list)
    if pol_indices is None:
        pol_indices = slice(None)
    if pol_summation:
        my_asserteq(len(pol_indices), 2)

    # Check if data column is available and get shape
    with ms_table(name) as t:
        nmspol = t.getcol(data_column, startrow=0, nrow=3).shape[2]
        nrow = t.nrows()
    logger.info("Measurement set visibilities:")
    logger.info(f"  shape: ({nrow}, {_ms_nchannels(name, spectral_window)}, {nmspol})")

    active_rows, active_channels = _first_pass(name, field, spectral_window, channels, pol_indices,
                                               pol_summation, ignore_flags)
    nrealrows, nrealchan = np.sum(active_rows), np.sum(active_channels)
    if nrealrows == 0 or nrealchan == 0:
        return None

    # Freq
    freq = _ms_channels(name, spectral_window)[active_channels]

    # Vis, wgt, (flags)
    fullwgt, weightcol = _determine_weighting(name)
    nchan = _ms_nchannels(name, spectral_window)
    with ms_table(name) as t:
        if pol_summation:
            npol = 1
        else:
            if pol_indices == slice(None):
                npol = nmspol
            else:
                npol = len(pol_indices)
        shp = (nrealrows, nrealchan, npol)
        vis = np.empty(shp, dtype=np.complex64)
        wgt = np.empty(shp, dtype=np.float32)

        # Read in data
        start, realstart = 0, 0
        while start < nrow:
            logger.debug(f"Second pass: {(start/nrow*100):.1f}%")
            stop = _ms_stop(start, nrow)
            realstop = realstart + np.sum(active_rows[start:stop])
            if realstop > realstart:
                allrows = stop - start == realstop - realstart

                # Weights
                twgt = t.getcol(weightcol, startrow=start, nrow=stop - start)[..., pol_indices]
                assert twgt.dtype == np.float32
                if not fullwgt:
                    twgt = np.repeat(twgt[:, None], nchan, axis=1)
                if not allrows:
                    twgt = twgt[active_rows[start:stop]]
                twgt = twgt[:, active_channels]

                # Vis
                tvis = t.getcol(data_column, startrow=start, nrow=stop - start)[..., pol_indices]
                assert tvis.dtype == np.complex64
                if not allrows:
                    tvis = tvis[active_rows[start:stop]]
                tvis = tvis[:, active_channels]

                # Flags
                tflags = _conditional_flags(t, start, stop, pol_indices, ignore_flags)
                if not allrows:
                    tflags = tflags[active_rows[start:stop]]
                tflags = tflags[:, active_channels]

                # Polarization summation
                assert twgt.ndim == tflags.ndim == 3
                assert tflags.dtype == bool
                if not ignore_flags:
                    twgt = twgt * (~tflags)
                if pol_summation:
                    assert twgt.shape[2] == 2
                    # Noise-weighted average
                    tvis = np.sum(twgt * tvis, axis=-1)[..., None]
                    twgt = np.sum(twgt, axis=-1)[..., None]
                    tvis /= twgt
                del tflags
                vis[realstart:realstop] = tvis
                wgt[realstart:realstop] = twgt

            start, realstart = stop, realstop
    logger.info("Selected:" + 10 * " ")
    logger.info(f"  shape: {vis.shape}")
    logger.info(f"  flagged: {(1.0-np.sum(wgt!=0)/wgt.size)*100:.1f} %")

    # UVW
    with ms_table(name) as t:
        uvw = np.ascontiguousarray(t.getcol("UVW")[active_rows])

    # Calibration info
    if with_calib_info:
        with ms_table(name) as t:
            ant1 = np.ascontiguousarray(t.getcol("ANTENNA1")[active_rows])
            ant2 = np.ascontiguousarray(t.getcol("ANTENNA2")[active_rows])
            time = np.ascontiguousarray(t.getcol("TIME")[active_rows])
    else:
        ant1 = ant2 = time = None

    # Pointing
    with ms_table(join(name, "POINTING")) as t:
        if t.nrows() == 0:
            ptg = None
        else:
            if t.nrows() == active_rows.size:
                ptg = np.empty((nrealrows, 1, 2), dtype=np.float64)
                start, realstart = 0, 0
                while start < nrow:
                    logger.debug(f"Second pass: {(start/nrow*100):.1f}%")
                    stop = _ms_stop(start, nrow)
                    realstop = realstart + np.sum(active_rows[start:stop])
                    if realstop > realstart:
                        allrows = stop - start == realstop - realstart
                        tptg = t.getcol("DIRECTION", startrow=start, nrow=stop - start)
                        tptg = tptg[active_rows[start:stop]]
                        ptg[realstart:realstop] = tptg
                    start, realstart = stop, realstop
            else:
                print(f"WARN: number of rows of POINTING table ({t.nrows()}) does not match number "
                      f"of rows of main table ({active_rows.size}). Ignore POINTING table...")
                ptg = None

    my_asserteq(wgt.shape, vis.shape)
    vis = np.ascontiguousarray(_ms2resolve_transpose(vis))
    wgt = np.ascontiguousarray(_ms2resolve_transpose(wgt))
    vis[wgt == 0] = 0.0
    return {"uvw": uvw, "ant1": ant1, "ant2": ant2, "time": time, "freq": freq, "vis": vis,
            "wgt": wgt, "ptg": ptg}


def _first_pass(ms, field, spectral_window, channels, pol_indices, pol_summation, ignore_flags):
    """Go through measurement set and determine which rows and which channels are active for a given
    field and a given spectral window.
    """
    fullwgt, weightcol = _determine_weighting(ms)
    nchan = _ms_nchannels(ms, spectral_window)
    with ms_table(ms) as t:
        nrow = t.nrows()
        active_rows = np.ones(nrow, dtype=bool)
        active_channels = np.zeros(nchan, dtype=bool)

        # Determine which subset of rows/channels we need to input
        start = 0
        while start < nrow:
            logger.debug(f"First pass: {(start/nrow*100):.1f}%")
            stop = _ms_stop(start, nrow)
            tflags = _conditional_flags(t, start, stop, pol_indices, ignore_flags)
            twgt = t.getcol(weightcol, startrow=start, nrow=stop - start)[..., pol_indices]

            tchslcflags = np.ones_like(tflags)
            tchslcflags[:, channels] = False
            tflags = np.logical_or(tflags, tchslcflags)

            if not fullwgt:
                twgt = np.repeat(twgt[:, None], nchan, axis=1)
            my_asserteq(twgt.ndim, 3)
            if pol_summation:
                tflags = np.any(tflags.astype(bool), axis=2)[..., None]
                twgt = np.sum(twgt, axis=2)[..., None]
            tflags[twgt == 0] = True

            # Select field and spectral window
            tfieldid = t.getcol("FIELD_ID", startrow=start, nrow=stop - start)
            tflags[tfieldid != field] = True
            tspw = t.getcol("DATA_DESC_ID", startrow=start, nrow=stop - start)
            tflags[tspw != spectral_window] = True

            # Inactive if all polarizations are flagged
            assert tflags.ndim == 3
            tflags = np.all(tflags, axis=2)
            active_rows[start:stop] = np.invert(np.all(tflags, axis=1))
            active_channels = np.logical_or(active_channels, np.invert(np.all(tflags, axis=0)))
            start = stop
    return active_rows, active_channels


def _ms_stop(start, nrow):
    """Compute sensible step size for going through the rows of a measurement set and return stop
    index.
    """
    step = max(1, nrow // 100)
    return min(nrow, start + step)


def _ms_nchannels(ms, spectral_window):
    """Return number of channels in a given spectral window of a measurement set.
    """
    return len(_ms_channels(ms, spectral_window))


def _ms_channels(ms, spectral_window):
    """Return frequencies of channels in a given spectral window of a measurement set.
    """
    with ms_table(join(ms, "SPECTRAL_WINDOW")) as t:
        freq = t.getcol("CHAN_FREQ", startrow=spectral_window, nrow=1)[0]
    my_asserteq(freq.ndim, 1)
    my_assert(len(freq) > 0)
    return freq


def ms_n_spectral_windows(ms):
    with ms_table(join(ms, "SPECTRAL_WINDOW")) as t:
        n_spectral_windows = t.nrows()
    return n_spectral_windows


def ms_n_fields(ms):
    with ms_table(join(ms, "FIELD")) as t:
        n = t.nrows()
    return n


def _conditional_flags(table, start, stop, pol_indices, ignore):
    tflags = table.getcol("FLAG", startrow=start, nrow=stop - start)[..., pol_indices]
    if ignore:
        tflags = np.zeros_like(tflags)
    return tflags


def _import_aux_table(ms, table_name, row=None, skip=[]):
    with ms_table(join(ms, table_name)) as t:
        keys = filter(lambda s: s not in skip, t.colnames())
        if row is None:
            dct = {kk: t.getcol(kk) for kk in keys}
        else:
            dct = {kk: t.getcol(kk, startrow=row, nrow=1) for kk in keys}
        aux_table = AuxiliaryTable(dct)
    return aux_table


def _source_and_field_table(ms, spectral_window, ifield):
    source_table = _import_aux_table(ms, "SOURCE", skip=["POSITION", "TRANSITION", "REST_FREQUENCY",
                                                         "SYSVEL", "SOURCE_MODEL", "PULSAR_ID"])
    field_table = _import_aux_table(ms, "FIELD", row=ifield)
    source_id = field_table["SOURCE_ID"][0]
    ind = np.where(np.logical_and(source_table["SOURCE_ID"] == source_id,
                                  np.logical_or(source_table["SPECTRAL_WINDOW_ID"] == spectral_window,
                                                source_table["SPECTRAL_WINDOW_ID"] == -1)))[0]
    if len(ind) == 0:
        return None
    elif len(ind) == 1:
        ind = ind[0]
    else:
        raise RuntimeError

    return {"SOURCE": source_table.row(ind), "FIELD": field_table}
