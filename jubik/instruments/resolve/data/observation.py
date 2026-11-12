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

import warnings

import numpy as np
import astropy.units as u

from ..conventions import OBSERVATION_FREQ_UNIT, OBSERVATION_UVW_UNIT
from ..util import (
    compare_attributes,
    is_single_precision,
)
from .antenna_positions import AntennaPositions
from .auxiliary_table import AuxiliaryTable
from .direction import Direction, Directions
from ....polarization import Polarization, PolarizationType


class BaseObservation:
    @property
    def vis_val(self):
        """Deprecated alias for `vis`.

        numpy.ndarray : Array that contains all data points including
        potentially flagged ones.  Shape: `(npol, nrow, nchan)`, dtype: `numpy.complexfloating`.
        """
        warnings.warn(
            "vis_val is deprecated; use vis instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.vis

    @property
    def vis(self):
        """numpy.ndarray : Array that contains all data points including
        potentially flagged ones.  Shape: `(npol, nrow, nchan)`, dtype: `numpy.complexfloating`.
        """
        return self._vis

    @property
    def weight_val(self):
        """Deprecated alias for `weight`.

        numpy.ndarray : Array that contains all weights, i.e. the diagonal of
        the inverse covariance. Shape: `(npol, nrow, nchan)`, dtype: `numpy.floating`.

        Note
        ----
        If an entry equals 0, this means that the corresponding data point is
        supposed to be ignored.
        """
        warnings.warn(
            "weight_val is deprecated; use weight instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.weight

    @property
    def weight(self):
        """numpy.ndarray : Array that contains all weights, i.e. the diagonal of
        the inverse covariance. Shape: `(npol, nrow, nchan)`, dtype: `numpy.floating`.

        Note
        ----
        If an entry equals 0, this means that the corresponding data point is
        supposed to be ignored.
        """
        return self._weight

    @property
    def freq(self):
        """numpy.ndarray: One-dimensional array that contains the observing
        frequencies in Hz as bare numerical values.
        Shape: ``(nchan,)``, dtype: ``np.float64``.
        """
        return self._freq

    @property
    def polarization(self) -> PolarizationType:
        """Polarization: Object that contains polarization information on the
        data set."""
        return PolarizationType.from_polarization_object(self._polarization)

    @property
    def legacy_polarization(self):
        """Polarization: Object that contains polarization information on the
        data set."""
        return self._polarization

    @property
    def polarization_labels(self):
        """tuple[str, ...]: Labels in visibility-axis order, including Stokes I."""
        return tuple(self.legacy_polarization.to_str_list()) or ("I",)

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

    def apply_flags(self, data):
        """Return non-flagged entries of a visibility-shaped array as a flattened 1D array."""
        return self.mask_operator(data)

    @property
    def flags_val(self):
        """Deprecated alias for `flags`.

        numpy.ndarray: True for bad visibilities."""
        warnings.warn(
            "flags_val is deprecated; use flags instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.flags

    @property
    def flags(self):
        """numpy.ndarray: True for bad visibilities."""
        return self._weight == 0.0

    @property
    def mask_val(self):
        """Deprecated alias for `mask`.

        numpy.ndarray: True for good visibilities."""
        warnings.warn(
            "mask_val is deprecated; use mask instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.mask

    @property
    def mask(self):
        """numpy.ndarray: True for good visibilities."""
        return self._weight > 0.0

    @property
    def mask_operator(self):
        """Fixed-index selection callable compatible with JAX autodiff and JIT.

        Construct this callable before tracing. Use ``jax.linear_transpose``
        when an explicit adjoint is needed.
        """
        indices = np.flatnonzero(self.mask)
        shape = self.vis.shape

        def select(data):
            if data.shape != shape:
                raise ValueError("Data shape must match the observation visibility shape.")
            return data.reshape(-1)[indices]

        return select

    def flags_to_nan(self):
        raise NotImplementedError

    def max_snr(self):
        """float: Maximum signal-to-noise ratio."""
        snr = np.abs(self.vis) * np.sqrt(self.weight)
        snr = self.apply_flags(snr)
        return np.max(snr)

    def fraction_useful(self):
        """float: Fraction of non-flagged data points."""
        return self.n_data_effective() / self._vis.size

    def n_data_effective(self):
        """int: Number of effective (i.e. non-flagged) data points."""
        return np.count_nonzero(self.mask)

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


class Observation(BaseObservation):
    """Provide an interface to an interferometric observation.

    This class contains all the data and information about an observation.
    It supports a single field (phase center) and a single spectral window.

    Parameters
    ----------
    antenna_positions : AntennaPositions
        Contains all information on antennas and baselines; UVW values are in metres.
    vis : numpy.ndarray
        Contains the measured visibilities. Shape (n_polarizations, n_rows, n_channels)
    weight : numpy.ndarray
        Contains the information from the WEIGHT or SPECTRUM_WEIGHT column.
        This is in many cases the inverse of the thermal noise covariance. Shape same as vis.
    polarization : Polarization
        Polarization information of the data set.
    freq : numpy.ndarray
        Contains the measured frequencies in Hz as bare numerical values.
        Shape (n_channels)
    auxiliary_tables : dict
        Dictionary of auxiliary tables. Default: None.

    Note
    ----
    vis and weight must have the same shape. Frequencies and UVW coordinates
    are stored as numerical arrays in Hz and metres, respectively. Readers must
    convert to this convention before constructing an Observation.
    """

    def __init__(
        self,
        antenna_positions,
        vis,
        weight,
        polarization: Polarization,
        freq,
        auxiliary_tables=None,
    ):
        nrows = len(antenna_positions)
        if not isinstance(polarization, Polarization):
            raise RuntimeError(
                f"Polarization metadata must be Polarization; got {type(polarization).__name__}."
            )
        if not isinstance(antenna_positions, AntennaPositions):
            raise RuntimeError(
                f"Antenna positions must be AntennaPositions; got {type(antenna_positions).__name__}."
            )
        if weight.shape != vis.shape:
            raise RuntimeError("Weights and visibilities must have matching shapes.")
        if vis.shape != (len(polarization), nrows, len(freq)):
            raise RuntimeError(
                "Visibilities must have axes (polarizations, rows, frequencies) "
                f"with shape {(len(polarization), nrows, len(freq))}, matching "
                "the polarization labels, antenna positions, and frequencies; "
                f"got shape {vis.shape}."
            )
        if nrows != vis.shape[1]:
            raise RuntimeError(
                f"Visibility axis 1 contains {vis.shape[1]} rows, but antenna "
                f"positions contain {nrows} rows; each visibility row must "
                "have a corresponding UVW coordinate."
            )
        if not (np.all(weight >= 0.0)):
            raise RuntimeError('Visibility weights must be nonnegative.')
        if not (np.all(np.isfinite(vis[weight > 0.0]))):
            raise RuntimeError('Unflagged visibilities must be finite.')
        if not (np.all(np.isfinite(weight))):
            raise RuntimeError('Visibility weights must be finite.')
        if not (np.all(np.diff(freq))):
            raise RuntimeError('Adjacent frequencies must differ.')

        if is_single_precision(vis.dtype) != is_single_precision(weight.dtype):
            raise RuntimeError("Visibilities and weights must use matching precision.")

        vis.flags.writeable = False
        weight.flags.writeable = False

        self._antpos = antenna_positions
        self._vis = vis
        self._weight = weight
        self._polarization = polarization
        self._freq = freq

        if auxiliary_tables is not None:
            if not isinstance(auxiliary_tables, dict):
                raise RuntimeError(
                    f"Auxiliary tables must be dict; got {type(auxiliary_tables).__name__}."
                )
            for kk, vv in auxiliary_tables.items():
                if not isinstance(vv, AuxiliaryTable):
                    raise RuntimeError(
                        f"Auxiliary table value must be AuxiliaryTable; got {type(vv).__name__}."
                    )
                if not isinstance(kk, str):
                    raise RuntimeError(
                        f"Table key must be str; got {type(kk).__name__}."
                    )
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

    def flags_to_nan(self):
        if self.fraction_useful() == 1.0:
            return self
        vis = self._vis.copy()
        vis[self.flags] = np.nan
        return Observation(
            self._antpos,
            vis,
            self._weight,
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

    # # TODO: Delete this functionality

    @property
    def nbaselines(self):
        return len(self.baselines())

    def baselines(self):
        return set((a1, a2) for a1, a2 in zip(self.ant1, self.ant2))

    @property
    def uvw(self):
        """Baseline coordinates in metres as a numerical array of shape (nrow, 3)."""
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

            if "NAME" in self._auxiliary_tables["FIELD"]:
                name = self._auxiliary_tables["FIELD"]["NAME"][0]
            else:
                name = ""

            return Direction(refdir[0], equinox, name)
        return None

    def direction_from_key(self, key):
        if self._auxiliary_tables is not None and "FIELD" in self._auxiliary_tables:
            equinox = (
                2000  # FIXME Figure out how to extract this from a measurement set
            )
            refdir = self._auxiliary_tables["FIELD"][key][0]

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
        """Return baselines in multiples of wavelengths, with shape (3, nrow, nfreq)."""
        frequencies = self.freq * OBSERVATION_FREQ_UNIT
        wavelengths = frequencies.to(OBSERVATION_UVW_UNIT, equivalencies=u.spectral())
        out = np.einsum("ij,k->jik", self.uvw, 1 / wavelengths.value)
        if out.shape != (3, self.nrow, self.nfreq):
            raise RuntimeError("Effective UVW coordinates must have shape (3, rows, frequencies).")
        return out

    def effective_uvwlen(self):
        """Return baseline lengths in multiples of wavelengths on the visibility domain."""
        frequencies = self.freq * OBSERVATION_FREQ_UNIT
        wavelengths = frequencies.to(OBSERVATION_UVW_UNIT, equivalencies=u.spectral())
        arr = np.outer(self.uvwlen(), 1 / wavelengths.value)
        arr = np.broadcast_to(arr[None], self._vis.shape)
        return arr

    def uvwlen(self):
        return np.linalg.norm(self.uvw, axis=1)

    def __str__(self):
        baseline_min = self.uvwlen().min() * OBSERVATION_UVW_UNIT
        baseline_max = self.uvwlen().max() * OBSERVATION_UVW_UNIT

        freq_min = self.freq.min() * OBSERVATION_FREQ_UNIT
        freq_max = self.freq.max() * OBSERVATION_FREQ_UNIT

        angular_scale_min = (
            freq_max.to(OBSERVATION_UVW_UNIT, equivalencies=u.spectral()) / baseline_max
        ).to_value(u.dimensionless_unscaled) * u.rad
        angular_scale_max = (
            freq_min.to(OBSERVATION_UVW_UNIT, equivalencies=u.spectral()) / baseline_min
        ).to_value(u.dimensionless_unscaled) * u.rad

        s = [
            f"Source name:\t\t{self.source_name}",
            f"Visibilities shape:\t{self.vis.shape}",
            f"# visibilities:\t{self.vis.size}",
            f"Frequency range:\t{freq_min.to_value(u.MHz):.3f} -- "
            f"{freq_max.to_value(u.MHz):.3f} MHz",
            "Polarizations:\t" + ", ".join(self.polarization_labels),
            f"Shortest baseline:\t{baseline_min.to_value(u.m):.1f} m -> "
            f"{angular_scale_max.to_value(u.deg):.3f} deg",
            f"Longest baseline:\t{baseline_max.to_value(u.m):.1f} m -> "
            f"{angular_scale_min.to_value(u.arcsec):.3f} arcsec",
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
    for index, observation in enumerate(args):
        if not isinstance(observation, Observation):
            raise RuntimeError(
                f"Argument {index} must be an Observation; got {type(observation).__name__}."
            )
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
    for index, observation in enumerate(args):
        if not isinstance(observation, Observation):
            raise RuntimeError(
                f"Argument {index} must be an Observation; got {type(observation).__name__}."
            )
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
    for index, observation in enumerate(args):
        if not isinstance(observation, Observation):
            raise RuntimeError(
                f"Argument {index} must be an Observation; got {type(observation).__name__}."
            )
    times = set()
    for oo in args:
        times = times | oo.antenna_positions.unique_times()
    return times
