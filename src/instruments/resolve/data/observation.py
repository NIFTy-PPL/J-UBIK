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
# Author: Philipp Arras

import nifty.cl as ift
import numpy as np

from ..constants import AS2RAD, DEG2RAD, SPEEDOFLIGHT
from ..util import (
    compare_attributes,
    my_assert,
    my_assert_isinstance,
    my_asserteq,
    is_single_precision,
)
from .antenna_positions import AntennaPositions
from .auxiliary_table import AuxiliaryTable
from .direction import Direction, Directions
from .polarization import Polarization


class BaseObservation:
    @property
    def _dom(self):
        pol_dom = self.polarization.space
        dom = [pol_dom] + [ift.UnstructuredDomain(ss) for ss in self._vis.shape[1:]]
        return ift.makeDomain(dom)

    @property
    def vis_val(self):
        """numpy.ndarray : Array that contains all data points including
        potentially flagged ones.  Shape: `(npol, nrow, nchan)`, dtype: `numpy.complexfloating`."""
        return self._vis

    @property
    def vis(self):
        """nifty.cl.Field : Field that contains all data points including
        potentially flagged ones.  Shape: `(npol, nrow, nchan)`, dtype: `numpy.complexfloating`."""
        return ift.makeField(self._dom, self._vis)

    @property
    def weight_val(self):
        """numpy.ndarray : Array that contains all weights, i.e. the diagonal of
        the inverse covariance. Shape: `(npol, nrow, nchan)`, dtype: `numpy.floating`.

        Note
        ----
        If an entry equals 0, this means that the corresponding data point is
        supposed to be ignored.
        """
        return self._weight

    @property
    def weight(self):
        """nifty.cl.Field : Field that contains all weights, i.e. the diagonal of
        the inverse covariance. Shape: `(npol, nrow, nchan)`, dtype: `numpy.floating`.

        Note
        ----
        If an entry equals 0, this means that the corresponding data point is
        supposed to be ignored.
        """
        return ift.makeField(self._dom, self._weight)

    @property
    def freq(self):
        """numpy.ndarray: One-dimensional array that contains the observing
        frequencies. Shape: `(nchan,), dtype: `np.float64`."""
        return self._freq

    @property
    def polarization(self):
        """Polarization: Object that contains polarization information on the
        data set."""
        return self._polarization

    @property
    def direction(self):
        """Direction: Object that contains direction information on the data
        set."""
        return self._direction

    @property
    def npol(self):
        """int: Number of polarizations present in the data set."""
        return self._vis.shape[0]

    @property
    def nrow(self):
        """int: Number of rows in the data set."""
        return self._vis.shape[1]

    @property
    def nfreq(self):
        """int: Number of observing frequencies."""
        return self._vis.shape[2]

    def apply_flags(self, field):
        """Apply flags to a given field.

        Parameters
        ----------
        field: nifty.cl.Field
            The field that is supposed to be flagged.

        Returns
        -------
        nifty.cl.Field
            Flagged field defined on a one-dimensional
            `nifty.cl.UnstructuredDomain`."""
        return self.mask_operator(field)

    @property
    def flags_val(self):
        """numpy.ndarray: True for bad visibilities."""
        return self._weight == 0.0

    @property
    def flags(self):
        """nifty.cl.Field: True for bad visibilities. May be used together with
        `ift.MaskOperator`."""
        return ift.makeField(self._dom, self._weight == 0.0)

    @property
    def mask_val(self):
        """numpy.ndarray: True for good visibilities."""
        return self._weight > 0.0

    @property
    def mask(self):
        """nifty.cl.Field: True for good visibilities."""
        return ift.makeField(self._dom, self._weight > 0.0)

    @property
    def mask_operator(self):
        """nifty.cl.MaskOperator: Nifty operator that can be used to extract all
        non-flagged data points from a field defined on `self.vis.domain`."""
        return ift.MaskOperator(self.flags)

    def flags_to_nan(self):
        raise NotImplementedError

    def max_snr(self):
        """float: Maximum signal-to-noise ratio."""
        snr = (self.vis * self.weight.sqrt()).abs()
        snr = self.apply_flags(snr)
        return np.max(snr.val.val)

    def fraction_useful(self):
        """float: Fraction of non-flagged data points."""
        return self.n_data_effective() / self._dom.size

    def n_data_effective(self):
        """int: Number of effective (i.e. non-flagged) data points."""
        return self.mask.s_sum()

    def save(self, file_name, compress):
        """Save observation object to disk

        Counterpart to :meth:`load`.

        Parameters
        ----------
        file_name : str
            File name of output file
        compress : bool
            Determine if output file shall be compressed or not. The compression
            algorithm built into numpy is used for this.
        """
        return NotImplementedError

    @staticmethod
    def load(file_name):
        """Load observation object from disk

        Counterpart to :meth:`save`.

        Parameters
        ----------
        file_name : str
            File name of the input file
        """
        return NotImplementedError

    def __getitem__(self, slc):
        return NotImplementedError

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if (
            self._vis.dtype != other._vis.dtype
            or self._weight.dtype != other._weight.dtype
        ):
            return False
        return compare_attributes(self, other, self._eq_attributes)

    def __hash__(self):
        # Since Observations are immutable objects this is okay
        return id(self)

    @property
    def antenna_names(self):
        return list(self.auxiliary_table("ANTENNA")["NAME"])


class SingleDishObservation(BaseObservation):
    """Provide an interface to single-dish observation.

    Parameters
    ----------
    pointings: Directions
        Contains all information on the observing directions.
    data: numpy.ndarray
        Contains the measured intensities. Shape (n_polarizations, n_rows,
        n_channels).
    weight : numpy.ndarray
        Contains the information from the WEIGHT or SPECTRUM_WEIGHT column.
        This is in many cases the inverse of the thermal noise covariance.
        Shape same as `data`.
    polarization : Polarization
        Polarization information of the data set.
    freq : numpy.ndarray
        Contains the measured frequencies. Shape (n_channels)
    """

    def __init__(self, pointings, data, weight, polarization, freq):
        my_assert_isinstance(pointings, Directions)
        my_assert_isinstance(polarization, Polarization)
        my_assert(data.dtype in [np.float32, np.float64])
        nrows = len(pointings)
        my_asserteq(weight.shape, data.shape)
        my_asserteq(data.shape, (len(polarization), nrows, len(freq)))
        my_asserteq(nrows, data.shape[1])

        data.flags.writeable = False
        weight.flags.writeable = False

        my_assert(np.all(weight >= 0.0))
        my_assert(np.all(np.isfinite(data)))
        my_assert(np.all(np.isfinite(weight)))

        self._pointings = pointings
        self._vis = data
        self._weight = weight
        self._polarization = polarization
        self._freq = freq

        self._eq_attributes = "_polarization", "_freq", "_pointings", "_vis", "_weight"

    def save(self, file_name, compress):
        p = self._pointings.to_list()
        dct = dict(
            vis=self._vis,
            weight=self._weight,
            freq=self._freq,
            polarization=self._polarization.to_list(),
            pointings0=p[0],
            pointings1=p[1],
        )
        f = np.savez_compressed if compress else np.savez
        f(file_name, **dct)

    @staticmethod
    def load(file_name):
        dct = dict(np.load(file_name))
        pol = Polarization.from_list(dct["polarization"])
        pointings = Directions.from_list([dct["pointings0"], dct["pointings1"]])
        return SingleDishObservation(
            pointings, dct["vis"], dct["weight"], pol, dct["freq"]
        )

    def __getitem__(self, slc):
        return SingleDishObservation(
            self._pointings[slc],
            self._vis[:, slc],
            self._weight[:, slc],
            self._polarization,
            self._freq,
        )

    @property
    def pointings(self):
        return self._pointings


class Observation(BaseObservation):
    """Provide an interface to an interferometric observation.

    This class contains all the data and information about an observation.
    It supports a single field (phase center) and a single spectral window.

    Parameters
    ----------
    antenna_positions : AntennaPositions
        Contains all information on antennas and baselines.
    vis : numpy.ndarray
        Contains the measured visibilities. Shape (n_polarizations, n_rows, n_channels)
    weight : numpy.ndarray
        Contains the information from the WEIGHT or SPECTRUM_WEIGHT column.
        This is in many cases the inverse of the thermal noise covariance. Shape same as vis.
    polarization : Polarization
        Polarization information of the data set.
    freq : numpy.ndarray
        Contains the measured frequencies. Shape (n_channels)
    auxiliary_tables : dict
        Dictionary of auxiliary tables. Default: None.

    Note
    ----
    vis and weight must have the same shape.
    """

    def __init__(
        self, antenna_positions, vis, weight, polarization, freq, auxiliary_tables=None
    ):
        nrows = len(antenna_positions)
        my_assert_isinstance(polarization, Polarization)
        my_assert_isinstance(antenna_positions, AntennaPositions)
        my_asserteq(weight.shape, vis.shape)
        my_asserteq(vis.shape, (len(polarization), nrows, len(freq)))
        my_asserteq(nrows, vis.shape[1])
        my_assert(np.all(weight >= 0.0))
        my_assert(np.all(np.isfinite(vis[weight > 0.0])))
        my_assert(np.all(np.isfinite(weight)))
        my_assert(np.all(np.diff(freq)))

        my_assert(
            not (is_single_precision(vis.dtype) ^ is_single_precision(weight.dtype))
        )

        vis.flags.writeable = False
        weight.flags.writeable = False

        self._antpos = antenna_positions
        self._vis = vis
        self._weight = weight
        self._polarization = polarization
        self._freq = freq

        if auxiliary_tables is not None:
            my_assert_isinstance(auxiliary_tables, dict)
            for kk, vv in auxiliary_tables.items():
                my_assert_isinstance(vv, AuxiliaryTable)
                my_assert_isinstance(kk, str)
        self._auxiliary_tables = auxiliary_tables

        self._eq_attributes = (
            "_polarization",
            "_freq",
            "_antpos",
            "_vis",
            "_weight",
            "_auxiliary_tables",
        )

    def save(self, file_name, compress):
        dct = dict(
            vis=self._vis,
            weight=self._weight,
            freq=self._freq,
            polarization=self._polarization.to_list(),
        )
        for ii, vv in enumerate(self._antpos.to_list()):
            if vv is None:
                vv = np.array([])
            dct[f"antpos{ii}"] = vv
        if self._auxiliary_tables is not None:
            for kk, auxtable in self._auxiliary_tables.items():
                for ii, elem in enumerate(auxtable.to_list()):
                    dct[f"auxtable_{kk}_{ii:>04}"] = elem
        f = np.savez_compressed if compress else np.savez
        f(file_name, **dct)

    @staticmethod
    def load(file_name, lo_hi_index=None):
        dct = np.load(file_name)
        antpos = []
        for ii in range(4):
            val = dct[f"antpos{ii}"]
            if val.size == 0:
                val = None
            antpos.append(val)

        # Load auxtables
        keys = set(kk[9:-5] for kk in dct.keys() if kk[:9] == "auxtable_")
        if len(keys) == 0:
            auxtables = None
        else:
            auxtables = {}
            for kk in keys:
                ii = 0
                inp = []
                while f"auxtable_{kk}_{ii:>04}" in dct.keys():
                    inp.append(dct[f"auxtable_{kk}_{ii:>04}"])
                    ii += 1
                auxtables[kk] = AuxiliaryTable.from_list(inp)
        # /Load auxtables

        pol = Polarization.from_list(dct["polarization"])
        vis = dct["vis"]
        wgt = dct["weight"]
        freq = dct["freq"]
        if lo_hi_index is not None:
            slc = slice(*lo_hi_index)
            # Convert view into its own array
            vis = vis[..., slc].copy()
            wgt = wgt[..., slc].copy()
            freq = freq[slc].copy()
        del dct
        antpos = AntennaPositions.from_list(antpos)
        return Observation(antpos, vis, wgt, pol, freq, auxtables)

    @staticmethod
    def legacy_load(file_name, lo_hi_index=None):
        """Provide potentially incomplete interface for loading legacy npz files."""
        dct = np.load(file_name)
        antpos = []
        for ii in range(4):
            val = dct[f"antpos{ii}"]
            if val.size == 0:
                val = None
            antpos.append(val)
        antpos = AntennaPositions.from_list(antpos)
        pol = Polarization.from_list(dct["polarization"])
        vis = dct["vis"]
        wgt = dct["weight"]
        freq = dct["freq"]
        if lo_hi_index is not None:
            slc = slice(*lo_hi_index)
            # Convert view into its own array
            vis = vis[..., slc].copy()
            wgt = wgt[..., slc].copy()
            freq = freq[slc].copy()
        del dct
        return Observation(antpos, vis, wgt, pol, freq)

    def flags_to_nan(self):
        if self.fraction_useful == 1.0:
            return self
        vis = self._vis.copy()
        vis[self.flags.val.val] = np.nan
        return Observation(
            self._antpos,
            vis,
            self._weight,
            self._polarization,
            self._freq,
            self._auxiliary_tables,
        )

    @staticmethod
    def load_mf(file_name, n_imaging_bands, comm=None):
        if comm is not None:
            my_assert(n_imaging_bands >= comm.Get_size())

        # Compute frequency ranges in data space
        global_freqs = np.load(file_name).get("freq")
        assert np.all(np.diff(global_freqs) > 0)
        my_assert(n_imaging_bands <= global_freqs.size)

        if comm is None:
            local_imaging_bands = range(n_imaging_bands)
        else:
            lo, hi = ift.utilities.shareRange(
                n_imaging_bands, comm.Get_size(), comm.Get_rank()
            )
            local_imaging_bands = range(lo, hi)
        full_obs = Observation.load(file_name)
        obs_list = []
        for ii in local_imaging_bands:
            slc = slice(
                *ift.utilities.shareRange(len(global_freqs), n_imaging_bands, ii)
            )
            obs_list.append(full_obs.get_freqs_by_slice(slc))
        nu0 = global_freqs.mean()
        return obs_list, nu0

    def to_double_precision(self):
        return Observation(
            self._antpos,
            self._vis.astype(np.complex128, casting="same_kind", copy=False),
            self._weight.astype(np.float64, casting="same_kind", copy=False),
            self._polarization,
            self._freq,
            self._auxiliary_tables,
        )

    def to_single_precision(self):
        return Observation(
            self._antpos,
            self._vis.astype(np.complex64, casting="same_kind", copy=False),
            self._weight.astype(np.float32, casting="same_kind", copy=False),
            self._polarization,
            self._freq,
            self._auxiliary_tables,
        )

    def is_single_precision(self):
        assert not (
            is_single_precision(self._weight.dtype)
            ^ is_single_precision(self._vis.dtype)
        )
        return is_single_precision(self._weight.dtype)

    def is_double_precision(self):
        assert not (
            is_single_precision(self._weight.dtype)
            ^ is_single_precision(self._vis.dtype)
        )
        return not is_single_precision(self._weight.dtype)

    @property
    def double_precision(self):
        return self._vis.dtype == np.complex128

    def __getitem__(self, slc, copy=False):
        # FIXME Do I need to change something in self._auxiliary_tables?
        ap = self._antpos[slc]
        vis = self._vis[:, slc]
        wgt = self._weight[:, slc]
        if copy:
            ap = ap.copy()
            vis = vis.copy()
            wgt = wgt.copy()
        return Observation(
            ap, vis, wgt, self._polarization, self._freq, self._auxiliary_tables
        )

    def get_freqs(self, frequency_list, copy=False):
        """Return observation that contains a subset of the present frequencies

        Parameters
        ----------
        frequency_list : list
            List of indices that shall be returned
        """
        mask = np.zeros(self.nfreq, dtype=bool)
        mask[frequency_list] = 1
        return self.get_freqs_by_slice(mask, copy)

    def get_freqs_by_slice(self, slc, copy=False):
        # FIXME Do I need to change something in self._auxiliary_tables?
        vis = self._vis[..., slc]
        wgt = self._weight[..., slc]
        freq = self._freq[slc]
        if copy:
            vis = vis.copy()
            wgt = wgt.copy()
            freq = freq.copy()
        return Observation(
            self._antpos, vis, wgt, self._polarization, freq, self._auxiliary_tables
        )

    def average_stokesi(self):
        # FIXME Do I need to change something in self._auxiliary_tables?
        if self.vis.domain[0].labels_eq("I"):
            return self
        my_asserteq(self._vis.shape[0], 2)
        my_asserteq(self._polarization.restrict_to_stokes_i(), self._polarization)
        vis = np.sum(self._weight * self._vis, axis=0)[None]
        wgt = np.sum(self._weight, axis=0)[None]
        invmask = wgt == 0.0
        vis /= wgt + np.ones_like(wgt) * invmask
        vis[invmask] = 0.0
        return Observation(
            self._antpos,
            vis,
            wgt,
            Polarization.trivial(),
            self._freq,
            self._auxiliary_tables,
        )

    def restrict_by_time(self, tmin, tmax, with_index=False):
        my_assert(all(np.diff(self.time) >= 0))
        start, stop = np.searchsorted(self.time, [tmin, tmax])
        ind = slice(start, stop)
        res = self[ind]
        if with_index:
            return res, ind
        return res

    def restrict_by_freq(self, fmin, fmax, with_index=False):
        my_assert(all(np.diff(self.freq) > 0))
        start, stop = np.searchsorted(self.freq, [fmin, fmax])
        ind = slice(start, stop)
        res = self.get_freqs_by_slice(ind)
        if with_index:
            return res, ind
        return res

    def restrict_to_stokesi(self):
        if self.vis.domain[0].labels == ("I",):
            return self
        # FIXME Do I need to change something in self._auxiliary_tables?
        ind = self._polarization.stokes_i_indices()
        vis = self._vis[ind]
        wgt = self._weight[ind]
        pol = self._polarization.restrict_to_stokes_i()
        return Observation(
            self._antpos, vis, wgt, pol, self._freq, self._auxiliary_tables
        )

    def restrict_to_polarization(self, pol_label):
        # FIXME Do I need to change something in self._auxiliary_tables?
        ind = self.vis.domain[0].label2index(pol_label)
        vis = self._vis[ind : ind + 1]
        wgt = self._weight[ind : ind + 1]
        pol = Polarization([9])  # FIXME!!!!
        return Observation(
            self._antpos, vis, wgt, pol, self._freq, self._auxiliary_tables
        )

    def restrict_to_autocorrelations(self):
        slc = self._antpos.ant1 == self._antpos.ant2
        return self[slc]

    def remove_autocorrelations(self):
        slc = self._antpos.ant1 != self._antpos.ant2
        return self[slc]

    def move_time(self, t0):
        # FIXME Do I need to change something in self._auxiliary_tables?
        antpos = self._antpos.move_time(t0)
        return Observation(
            antpos,
            self._vis,
            self._weight,
            self._polarization,
            self._freq,
            self._auxiliary_tables,
        )

    def time_average(self, list_of_timebins):
        # FIXME check that timebins do not overlap
        # time, ant1, ant2
        ts = self._antpos.time
        row_to_bin_map = np.empty(ts.shape)
        row_to_bin_map[:] = np.nan

        for ii, (lo, hi) in enumerate(list_of_timebins):
            ind = np.logical_and(ts >= lo, ts < hi)
            assert np.all(np.isnan(row_to_bin_map[ind]))
            row_to_bin_map[ind] = ii
        assert np.all(np.diff(row_to_bin_map) >= 0)

        assert np.all(~np.isnan(row_to_bin_map))
        row_to_bin_map = row_to_bin_map.astype(int)

        ant1 = self._antpos.ant1
        ant2 = self._antpos.ant2
        assert np.max(ant1) < 1000
        assert np.max(ant2) < 1000
        assert np.max(row_to_bin_map) < np.iinfo(np.dtype("int64")).max / 1000000
        atset = np.array(list(set(zip(ant1, ant2, row_to_bin_map))))
        atset = atset[np.lexsort(atset.T)]
        atset = tuple(map(tuple, atset))
        dct = {aa: ii for ii, aa in enumerate(atset)}
        dct_inv = {yy: xx for xx, yy in dct.items()}
        masterindex = np.array(
            [dct[(a1, a2, tt)] for a1, a2, tt in zip(ant1, ant2, row_to_bin_map)]
        )

        vis, wgt = self.vis.val.val, self.weight.val.val
        new_vis = np.empty((self.npol, len(atset), self.nfreq), dtype=self.vis.dtype)
        new_wgt = np.empty((self.npol, len(atset), self.nfreq), dtype=self.weight.dtype)
        for pol in range(self.npol):
            for freq in range(self.nfreq):
                enum = np.bincount(
                    masterindex, weights=vis[pol, :, freq].real * wgt[pol, :, freq]
                )
                enum = enum + 1j * np.bincount(
                    masterindex, weights=vis[pol, :, freq].imag * wgt[pol, :, freq]
                )
                denom = np.bincount(masterindex, weights=wgt[pol, :, freq])
                if np.min(denom) == 0.0:
                    raise ValueError("Time bin with total weight 0. detected.")
                new_vis[pol, :, freq] = enum / denom
                new_wgt[pol, :, freq] = denom

        new_uvw = np.empty((len(atset), 3), dtype=self._antpos.uvw.dtype)
        new_times = np.empty(len(atset), dtype=self._antpos.time.dtype)
        new_uvw[()] = new_times[()] = np.nan
        denom = np.bincount(masterindex)
        # Assumption: Uvw value for averaged data is average of uvw values of finely binned data
        for ii in range(3):
            new_uvw[:, ii] = (
                np.bincount(masterindex, weights=self._antpos.uvw[:, ii]) / denom
            )
        new_times = np.bincount(
            row_to_bin_map, weights=self._antpos.time
        ) / np.bincount(row_to_bin_map)
        assert np.sum(np.isnan(new_uvw)) == 0
        assert np.sum(np.isnan(new_times)) == 0

        new_times = new_times[np.array([dct_inv[ii][2] for ii in range(len(atset))])]
        assert np.all(np.diff(new_times) >= 0)

        new_ant1 = np.array([dct_inv[ii][0] for ii in range(len(atset))])
        new_ant2 = np.array([dct_inv[ii][1] for ii in range(len(atset))])
        ap = AntennaPositions(new_uvw, new_ant1, new_ant2, new_times)
        return Observation(
            ap, new_vis, new_wgt, self._polarization, self._freq, self._auxiliary_tables
        )

    def flag_baseline(self, ant1_index, ant2_index):
        ant1 = self.antenna_positions.ant1
        ant2 = self.antenna_positions.ant2
        if ant1 is None or ant2 is None:
            raise RuntimeError(
                "The calibration information needed for flagging a baseline is not "
                "available. Please import the measurement set with "
                "`with_calib_info=True`."
            )
        assert np.all(ant1 < ant2)
        ind = np.logical_and(ant1 == ant1_index, ant2 == ant2_index)
        wgt = self._weight.copy()
        wgt[:, ind] = 0.0
        antenna_names = self.auxiliary_table("ANTENNA")["STATION"]
        ant1_name = antenna_names[ant1_index]
        ant2_name = antenna_names[ant2_index]
        if np.sum(ind) > 0:
            print(
                f"INFO: Flag baseline {ant1_name}-{ant2_name}, {np.sum(ind)}/{self.nrow} rows flagged."
            )
        return Observation(
            self._antpos,
            self._vis,
            wgt,
            self._polarization,
            self._freq,
            self._auxiliary_tables,
        )

    def flag_station(self, ant_index):
        ant1 = self.antenna_positions.ant1
        ant2 = self.antenna_positions.ant2
        if ant1 is None or ant2 is None:
            raise RuntimeError(
                "The calibration information needed for flagging a baseline is not "
                "available. Please import the measurement set with "
                "`with_calib_info=True`."
            )
        assert np.all(ant1 < ant2)
        ind = np.logical_or(ant1 == ant_index, ant2 == ant_index)
        wgt = self._weight.copy()
        wgt[:, ind] = 0.0
        if np.sum(ind) > 0:
            ant_name = self.auxiliary_table("ANTENNA")["STATION"][ant_index]
            print(
                f"INFO: Flag station {ant_name}, {np.sum(ind)}/{self.nrow} rows flagged."
            )
        return Observation(
            self._antpos,
            self._vis,
            wgt,
            self._polarization,
            self._freq,
            self._auxiliary_tables,
        )

    @property
    def nbaselines(self):
        return len(self.baselines())

    def baselines(self):
        return set((a1, a2) for a1, a2 in zip(self.ant1, self.ant2))

    @property
    def uvw(self):
        return self._antpos.uvw

    @property
    def antenna_positions(self):
        return self._antpos

    @property
    def ant1(self):
        return self._antpos.ant1

    @property
    def ant2(self):
        return self._antpos.ant2

    @property
    def time(self):
        return self._antpos.time

    @property
    def direction(self) -> Direction | None:
        if self._auxiliary_tables is not None and "FIELD" in self._auxiliary_tables:
            equinox = (
                2000  # FIXME Figure out how to extract this from a measurement set
            )
            refdir = self._auxiliary_tables["FIELD"]["REFERENCE_DIR"][0]
            my_asserteq(refdir.shape[0] == 0)

            if "NAME" in self._auxiliary_tables["FIELD"]:
                name = self._auxiliary_tables["FIELD"]["NAME"][0]
            else:
                name = ""

            return Direction(refdir[0], equinox, name)
        return None

    def direction_from_key(self, key) -> Direction | None:
        if self._auxiliary_tables is not None and "FIELD" in self._auxiliary_tables:
            equinox = (
                2000  # FIXME Figure out how to extract this from a measurement set
            )
            refdir = self._auxiliary_tables["FIELD"][key][0]
            my_asserteq(refdir.shape[0] == 0)

            if "NAME" in self._auxiliary_tables["FIELD"]:
                name = self._auxiliary_tables["FIELD"]["NAME"][0]
            else:
                name = ""

            return Direction(refdir[0], equinox, name)
        return None

    def auxiliary_table(self, name):
        return self._auxiliary_tables[name]

    @property
    def source_name(self):
        if self._auxiliary_tables is None:
            return None
        if "FIELD" in self._auxiliary_tables:
            return self._auxiliary_tables["FIELD"]["NAME"][0]
        raise NotImplementedError("FIELD subtable not available.")

    @property
    def station_names(self):
        """The index of the resulting list is the same as in self.ant1 or self.ant2."""
        if "ANTENNA" in self._auxiliary_tables:
            tab = self._auxiliary_tables["ANTENNA"]
            return [f"{a} {b}" for a, b in zip(tab["NAME"], tab["STATION"])]
        raise NotImplementedError("ANTENNA subtable not available.")

    def antenna_name2index(self, name):
        raise NotImplementedError

    def antenna_index2name(self, index):
        raise NotImplementedError

    @property
    def antenna_coordinates(self):
        # FIXME Merge this into AntennaPositions
        if "ANTENNA" not in self._auxiliary_tables:
            return None
        return self._auxiliary_tables["ANTENNA"]["POSITION"]

    def effective_uvw(self):
        out = np.einsum("ij,k->jik", self.uvw, self._freq / SPEEDOFLIGHT)
        my_asserteq(out.shape, (3, self.nrow, self.nfreq))
        return out

    def effective_uvwlen(self):
        arr = np.outer(self.uvwlen(), self._freq / SPEEDOFLIGHT)
        arr = np.broadcast_to(arr[None], self._dom.shape)
        return ift.makeField(self._dom, arr)

    def uvwlen(self):
        return np.linalg.norm(self.uvw, axis=1)

    def __str__(self):
        short0 = self.uvwlen().min()
        long0 = self.uvwlen().max()

        short1 = 1 / (short0 * self.freq.min() / SPEEDOFLIGHT)
        long1 = 1 / (long0 * self.freq.max() / SPEEDOFLIGHT)

        s = [
            f"Source name:\t\t{self.source_name}",
            f"Visibilities shape:\t{self.vis.shape}",
            f"# visibilities:\t{self.vis.size}",
            f"Frequency range:\t{self.freq.min() * 1e-6:.3f} -- {self.freq.max() * 1e-6:.3f} MHz",
            "Polarizations:\t" + ", ".join(self.vis.domain[0].labels),
            f"Shortest baseline:\t{short0:.1f} m -> {short1 / DEG2RAD:.3f} deg",
            f"Longest baseline:\t{long0:.1f} m -> {long1 / AS2RAD:.3f} arcsec",
        ]
        flagged = 1 - self.fraction_useful()
        if flagged == 0.0:
            s += ["Flagged:\t\tNone"]
        else:
            s += [f"Flagged:\t\t{flagged * 100:.1f}%"]
        return "\n".join(["Observation:"] + [f"  {ss}" for ss in s])


def tmin_tmax(*args):
    """Compute beginning and end time of list of observations.

    Parameters
    ----------
    args : Observation or list of Observation

    Returns
    -------
    mi, ma : tuple of float
        first and last measurement time point
    """
    my_assert_isinstance(*args, Observation)
    mi = min([np.min(aa.antenna_positions.time) for aa in args])
    ma = max([np.max(aa.antenna_positions.time) for aa in args])
    return mi, ma


def unique_antennas(*args):
    """Compute set of antennas of list of observations

    Parameters
    ----------
    args : Observation or list of Observation

    Returns
    -------
    set
        Set of antennas
    """
    my_assert_isinstance(*args, Observation)
    antennas = set()
    for oo in args:
        antennas = antennas | oo.antenna_positions.unique_antennas()
    return antennas


def unique_times(*args):
    """Compute set of time stamps of list of observations

    Parameters
    ----------
    args : Observation or list of Observation

    Returns
    -------
    set
        Set of time stamps
    """
    my_assert_isinstance(*args, Observation)
    times = set()
    for oo in args:
        times = times | oo.antenna_positions.unique_times()
    return times
