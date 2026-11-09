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

import jax.numpy as jnp


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
    """Store ordered polarization metadata using CASA/casacore identifiers.

    Codes 1--4 denote I, Q, U, V; 5--8 denote RR, RL, LR, LL; and
    9--12 denote XX, XY, YX, YY. These identifiers are not array positions.

    Parameters
    ----------
    indices : sequence of int
        Up to four polarization identifiers. An empty sequence is the legacy
        representation of a single Stokes I component: its length is one,
        although ``to_list()`` and ``to_str_list()`` return empty lists.
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
    """Named ordered polarization combinations supported by grids/observations.

    Values contain labels, not CASA identifiers. The matrix constructors also
    accept label sequences for subsets or orderings absent from this enum.
    """

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

    # Polarized subsets; reversed forms preserve existing data-axis order.
    Q_U = ("Q", "U")
    U_Q = ("U", "Q")
    U_V = ("U", "V")
    V_U = ("V", "U")
    RL_LR = ("RL", "LR")
    LR_RL = ("LR", "RL")
    XY_YX = ("XY", "YX")
    YX_XY = ("YX", "XY")

    # Full polarization
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
        """Shape of the polarization axis, ``(number_of_components,)``."""
        return (len(self),)

    def __len__(self):
        return len(self.value)


# Each row contains the coefficients expressing a polarization product as a
# linear combination of the reference Stokes vector (I, Q, U, V). For example,
# RR = I + V, RL = Q + iU, XX = I + Q, and XY = U + iV. These relations also
# apply to complex Stokes visibilities; no complex conjugation is implied.
# We use the radio flux normalization without a factor of 1/2: an unpolarized
# source has RR = LL = XX = YY = I and zero cross-hand products.
# The Stokes rows themselves select individual components. Matrix constructors
# select/order these rows for the outputs and use the same definitions for
# exact basis conversions and forward models with omitted Stokes set to zero.
_POLARIZATION_ROWS = {
    "I": (1, 0, 0, 0),
    "Q": (0, 1, 0, 0),
    "U": (0, 0, 1, 0),
    "V": (0, 0, 0, 1),
    "RR": (1, 0, 0, 1),
    "RL": (0, 1, 1j, 0),
    "LR": (0, 1, -1j, 0),
    "LL": (1, 0, 0, -1),
    "XX": (1, 1, 0, 0),
    "XY": (0, 0, 1, 1j),
    "YX": (0, 0, 1, -1j),
    "YY": (1, -1, 0, 0),
}

# Complete bases define matrix/vector order; subsets retain this relative order.
_POLARIZATION_BASES = (
    ("I", "Q", "U", "V"),
    ("RR", "RL", "LR", "LL"),
    ("XX", "XY", "YX", "YY"),
)


def _polarization_labels(polarization):
    labels = tuple(
        polarization.value
        if isinstance(polarization, PolarizationType)
        else polarization
    )
    if not labels or (len(set(labels)) != len(labels)):
        raise ValueError("Polarization labels must be nonempty and unique.")
    unknown = set(labels).difference(_POLARIZATION_ROWS)
    if unknown:
        raise ValueError(f"Unknown polarization labels: {sorted(unknown)}")
    return labels


def polarization_matrix(inp_pol, out_pol, *, assume_missing_stokes_zero=False):
    """Construct a polarization conversion matrix through full Stokes.

    Parameters
    ----------
    inp_pol, out_pol : PolarizationType or sequence of str
        Input and output labels in their data-axis order. Inputs must belong
        to one Stokes, circular, or linear basis. Subsets and permutations
        are supported.
    assume_missing_stokes_zero : bool, optional
        False (default) requires exact conversion: every output must be
        determined by the available inputs. True requires Stokes input and
        treats omitted Stokes components as zero for model predictions.
        For example, an I-only model then predicts RR = LL = I and zero
        cross hands, whereas exact I -> RR conversion is underdetermined.

    Returns
    -------
    jax.Array
        Matrix of shape (n_output, n_input). Rows and columns follow the
        supplied output and input order. Construct and validate this matrix
        outside JIT, or use build_polarization_conversion before compilation.
    """
    inputs = _polarization_labels(inp_pol)
    outputs = _polarization_labels(out_pol)
    basis = next(
        (basis for basis in _POLARIZATION_BASES if set(inputs) <= set(basis)), None
    )
    if basis is None:
        raise ValueError("Input products must belong to one polarization basis.")

    if assume_missing_stokes_zero and basis != _POLARIZATION_BASES[0]:
        raise ValueError(
            "The input polarization must be Stokes when assuming missing components zero."
        )

    # Construct the complete input-basis -> Stokes -> requested-output map.
    # For F = stokes_to_basis, orthogonal rows give F F^H = D, where D
    # contains squared row norms (1 for Stokes, 2 otherwise). Thus
    # F^{-1} = F^H D^{-1}, since F F^H D^{-1} = I.
    stokes_to_basis = jnp.array(
        [_POLARIZATION_ROWS[label] for label in basis], dtype=complex
    )
    # The complete Stokes basis has unit row norms; both correlation bases
    # have squared row norms of 2, so these are the only two possible forms.
    inverse_row_norm = 1 if basis == _POLARIZATION_BASES[0] else 0.5

    basis_to_stokes = inverse_row_norm * stokes_to_basis.conj().T
    # Build all output rows in canonical order before selecting/permuting at
    # the boundary. The supplied input order still describes actual data axes.
    canonical_outputs = tuple(label for label in _POLARIZATION_ROWS if label in outputs)
    stokes_to_output = jnp.array(
        [_POLARIZATION_ROWS[label] for label in canonical_outputs], dtype=complex
    )
    full_matrix = stokes_to_output @ basis_to_stokes

    # Discarding a missing product is exact only when none of the requested
    # outputs depend on it. Forward models explicitly assume missing Stokes
    # zero instead; exact data conversion must reject that assumption.
    if not assume_missing_stokes_zero:
        missing_indices = [i for i, label in enumerate(basis) if label not in inputs]

        if not jnp.allclose(
            full_matrix[:, jnp.asarray(missing_indices, dtype=int)],
            0,
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Polarization {inputs} cannot determine {outputs} "
                "without assumptions about missing components."
            )
    output_indices = jnp.array([canonical_outputs.index(label) for label in outputs])
    input_indices = jnp.array([basis.index(label) for label in inputs])
    return full_matrix[jnp.ix_(output_indices, input_indices)]


def build_polarization_conversion(
    inp_pol, out_pol, *, assume_missing_stokes_zero=False
):
    """Build a JAX callable applying polarization conversion on axis zero.

    Construct outside JIT: metadata validation and matrix construction happen
    once. The returned callable can be jitted, differentiated, or composed
    with a forward model. Trailing row/channel/spatial axes are unchanged.
    See polarization_matrix for the exact and zero-filled Stokes model modes.
    """
    matrix = polarization_matrix(
        inp_pol, out_pol, assume_missing_stokes_zero=assume_missing_stokes_zero
    )

    def convert(visibilities):
        dtype = jnp.result_type(visibilities, 1j)
        return jnp.tensordot(matrix.astype(dtype), visibilities, axes=([1], [0]))

    return convert
