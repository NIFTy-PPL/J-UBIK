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

import numpy as np

from ..util import compare_attributes


class AntennaPositions:
    """Summarizes all information on antennas and baselines. If calibration is
    performed this class stores also antenna indices and time information.
    For imaging only this is not necessary.

    Parameters
    ----------
    uvw : numpy.ndarray
        two-dimensional array of shape (n_rows, 3)
    ant1 : numpy.ndarray(dtype=int)
        one-dimensional array of shape (n_rows,)
    ant2 : numpy.ndarray(dtype=int)
        two-dimensional array of shape (n_rows,)
    time : float
        time of measurement
    """

    def __init__(self, uvw, ant1=None, ant2=None, time=None):
        if ant1 is None:
            if ant2 is not None or time is not None:
                raise RuntimeError(
                    "Antenna 2 indices and times must be absent when antenna 1 indices are absent."
                )
            if uvw.ndim != 2:
                raise RuntimeError("UVW coordinates must be two-dimensional.")
            if uvw.shape[1] != 3:
                raise RuntimeError("UVW coordinates must have three columns.")
        else:
            if not (ant1.shape == ant2.shape == time.shape):
                raise RuntimeError(
                    "Antenna indices and times must have matching shapes to have consistent row entries."
                )
            if uvw.shape != (ant1.size, 3):
                raise RuntimeError(
                    "UVW coordinates must have shape (number of data rows, 3)."
                )
            if not (np.issubdtype(ant1.dtype, np.integer)):
                raise RuntimeError('Antenna 1 indices must have an integer dtype.')
            if not (np.issubdtype(ant2.dtype, np.integer)):
                raise RuntimeError('Antenna 2 indices must have an integer dtype.')
            if not (np.issubdtype(time.dtype, np.floating)):
                raise RuntimeError('Times must have a floating point dtype.')
        if not (np.issubdtype(uvw.dtype, np.floating)):
            raise RuntimeError('UVW coordinates must have a floating point dtype.')
        self._uvw, self._time = uvw, time
        self._ant1, self._ant2 = ant1, ant2
        self._t0 = None

    @property
    def only_imaging(self):
        return self._ant1 is None

    def to_list(self):
        return [self._uvw, self._ant1, self._ant2, self._time]

    def unique_antennas(self):
        if self.only_imaging:
            raise RuntimeError
        return set(np.unique(self._ant1)) | set(np.unique(self._ant2))

    def unique_times(self):
        if self.only_imaging:
            raise RuntimeError
        return set(np.unique(self._time))

    @staticmethod
    def from_list(lst):
        return AntennaPositions(*lst)

    def move_time(self, t0):
        if self.only_imaging:
            raise RuntimeError
        return AntennaPositions(self._uvw, self._ant1, self._ant2, self._time + t0)

    def __eq__(self, other):
        if not isinstance(other, AntennaPositions):
            return False
        return compare_attributes(self, other, ("_uvw", "_time", "_ant1", "_ant2"))

    def __len__(self):
        return self._uvw.shape[0]

    def __getitem__(self, slc):
        if self.only_imaging:
            return AntennaPositions(self._uvw[slc])
        return AntennaPositions(
            self._uvw[slc], self._ant1[slc], self._ant2[slc], self._time[slc]
        )

    @property
    def uvw(self):
        return self._uvw

    @property
    def time(self):
        return self._time

    @property
    def ant1(self):
        return self._ant1

    @property
    def ant2(self):
        return self._ant2

    def extract_baseline(self, antenna1, antenna2, data):
        """Extract entries on in the data array that belongs to a given baseline.

        Parameters
        ----------
        antenna1: int
            Antenna index of the first antenna that is selected.
        antenna2: int
            Antenna index of the second antenna that is selected.
        data: numpy.ndarray
            Data array. Shape `(n_pol, n_row, n_freq)`. `n_row` must equal
            `len(self)`.

        Returns
        -------
        numpy.ndarray
            Entries of `data` that correspond to the selected baseline. Shape
            `(n_pol, n_time, n_freq)`.
        """
        if np.any(self.ant1 > self.ant2):
            raise RuntimeError("This algorithm assumes ant1<ant2.")
        ut = np.sort(np.array(list(self.unique_times())))

        npol, nrow, nfreq = data.shape
        if nrow != len(self):
            raise RuntimeError("Data row count must match the antenna positions.")
        if not isinstance(antenna1, int):
            raise RuntimeError(
                f"Antenna 1 index must be int; got {type(antenna1).__name__}."
            )
        if not isinstance(antenna2, int):
            raise RuntimeError(
                f"Antenna 2 index must be int; got {type(antenna2).__name__}."
            )

        # Select by antenna labels
        ind = np.logical_and(self.ant1 == antenna1, self.ant2 == antenna2)
        data = data[:, ind]
        tt = self.time[ind]

        # Sort by time
        ind2 = np.argsort(tt)
        data = data[:, ind2]
        tt = tt[ind2]

        if tt.size != ut.size:
            out = np.empty((npol, ut.size, nfreq), dtype=data.dtype)
            out[:] = np.nan
            out[:, np.searchsorted(ut, tt)] = data
        elif np.array_equal(tt, ut):
            out = data
        else:
            raise RuntimeError
        return out
