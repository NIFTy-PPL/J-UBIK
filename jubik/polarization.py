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
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras, Jokob Roth, Julian Rüstig

from typing import Iterable, Union
from enum import Enum, IntEnum

class PolarizationCode(IntEnum):
    """Supported CASA/casacore polarization identifiers.

    This enum contains the Stokes and correlation products supported by J-UBIK,
    not the full CASA/casacore enumeration.

    Look up a label with ``PolarizationCode(code).name`` and an integer
    identifier with ``PolarizationCode[label].value``.
    """

    I = 1
    Q = 2
    U = 3
    V = 4
    RR = 5
    RL = 6
    LR = 7
    LL = 8
    XX = 9
    XY = 10
    YX = 11
    YY = 12


class Polarization:
    """Stores polarization information of a data set and is used for
    translating between indices (5, 6, 7, 8, ...) and strings
    ("RR", "RL", "LR", "LL", ...).

    Parameters
    ----------
    indices : tuple of polarization indices.
        Takes integer values between 5 and including 12.
    """

    def __init__(self, indices):
        self._ind = tuple(indices)
        assert len(self._ind) <= 4
        self._trivial = len(indices) == 0

    @staticmethod
    def trivial():
        return Polarization([])

    def restrict_to_stokes_i(self):
        inds = (8, 5) if self.circular() else (9, 12)
        return Polarization(inds)

    def restrict_by_name(self, lst):
        return Polarization([PolarizationCode[ss].value for ss in lst])

    def circular(self):
        if self._trivial:
            raise RuntimeError
        if set(self._ind) <= set([5, 6, 7, 8]):
            return True
        if set(self._ind) <= set([9, 10, 11, 12]):
            return False
        raise RuntimeError

    def has_crosshanded(self):
        if len(set(self._ind) & set([6, 7, 10, 11])) > 0:
            return True
        return False

    def stokes_i_indices(self):
        if self._trivial:
            raise RuntimeError
        keys = ["LL", "RR"] if self.circular() else ["XX", "YY"]
        return [self._ind.index(PolarizationCode[kk].value) for kk in keys]

    def __len__(self):
        if self._trivial:
            return 1
        return len(self._ind)

    def to_list(self):
        return list(self._ind)

    def to_str_list(self):
        return [PolarizationCode(ii).name for ii in self._ind]

    @property
    def labels(self):
        """Ordered polarization labels, including ``I`` for trivial polarization.

        Unlike ``space``, accessing labels does not require classic NIFTy.
        """
        return tuple(self.to_str_list()) or ("I",)

    @property
    def space(self):
        """Legacy classic NIFTy domain; use ``labels`` for label-only access.

        The domain/operator replacement belongs to the polarization rework.
        """
        from .instruments.resolve.polarization_space import PolarizationSpace

        return PolarizationSpace(self.labels)


    @staticmethod
    def from_list(lst):
        return Polarization(lst)

    def __eq__(self, other):
        if not isinstance(other, Polarization):
            return False
        return self._ind == other._ind

    def __repr__(self):
        return f"Polarization({self._ind})"


class PolarizationType(Enum):
    # Total intensity
    I = ("I",)
    RR = ("RR",)
    LL = ("LL",)
    XX = ("XX",)
    YY = ("YY",)
    RR_LL = ("RR", "LL")
    LL_RR = ("LL", "RR")
    XX_YY = ("XX", "YY")
    YY_XX = ("YY", "XX")

    # Polarized
    IQUV = ("I", "Q", "U", "V")
    RR_RL_LR_LL = ("RR", "RL", "LR", "LL")
    XX_XY_YX_YY = ("XX", "XY", "YX", "YY")

    @property
    def is_single_feed(self):
        if self in (self.I, self.RR, self.LL, self.XX, self.YY):
            return True
        else:
            return False

    @classmethod
    def from_polarization_object(
        cls, polarization_object: Polarization
    ) -> "PolarizationType":
        typ = tuple(polarization_object.to_str_list())

        if typ == ():
            typ = ("I",)

        return cls(typ)

    def get_legacy_polarization(self):
        if self == self.I:
            indices = []
        else:
            indices = [PolarizationCode[p].value for p in self.value]
        return Polarization(indices)

    @property
    def shape(self):
        """Shape of the grid. (spectral, spatial)"""
        return (len(self),)

    def __len__(self):
        return len(self.value)
