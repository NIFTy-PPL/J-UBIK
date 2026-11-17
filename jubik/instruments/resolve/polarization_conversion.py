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
# Author: Philipp Arras, Jakob Knollmüller, Jokob Roth, Julian Rüstig

import jax.numpy as jnp

from ...polarization import Polarization, PolarizationType

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
    if isinstance(polarization, PolarizationType):
        polarization = polarization.value
    elif isinstance(polarization, Polarization):
        polarization = polarization.labels
    elif isinstance(polarization, str):
        polarization = (polarization,)
    labels = tuple(polarization)
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
    inp_pol, out_pol : PolarizationType, Polarization, or labels
        Input and output labels in their data-axis order. Inputs must belong
        to one Stokes, circular, or linear basis. Subsets and permutations
        are supported.
    assume_missing_stokes_zero : bool, optional
        False (default) requires exact conversion: every output must be
        determined by the available inputs. True requires Stokes input and
        treats omitted Stokes components as zero for model predictions.
        For example, an I-only model then predicts RR = LL = I and zero
        cross hands, whereas exact I -> RR conversion is underdetermined.
        Raises ValueError if none of the inputs contribute to the outputs.

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
    matrix = full_matrix[jnp.ix_(output_indices, input_indices)]
    if assume_missing_stokes_zero and not jnp.any(matrix != 0):
        raise ValueError(
            f"Polarization {inputs} does not contribute to {outputs}: "
            "the conversion matrix is all zero."
        )
    return matrix


def build_polarization_conversion(
    inp_pol, out_pol, *, assume_missing_stokes_zero=False, axis=0
):
    """Build a JAX callable applying conversion on the selected axis.

    Construct outside JIT: metadata validation and matrix construction happen
    once. The returned callable can be jitted, differentiated, or composed
    with a forward model. All axes except the polarization axis are unchanged.
    The default axis is zero; other axes support legacy layouts.
    See polarization_matrix for the exact and zero-filled Stokes model modes.
    """
    matrix = polarization_matrix(
        inp_pol, out_pol, assume_missing_stokes_zero=assume_missing_stokes_zero
    )

    def convert(visibilities):
        dtype = jnp.result_type(visibilities, 1j)
        result = jnp.tensordot(matrix.astype(dtype), visibilities, axes=([1], [axis]))
        return jnp.moveaxis(result, 0, axis)

    return convert
