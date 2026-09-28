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
# Copyright(C) 2026 Max-Planck-Society
# Author: Julian Rüstig

"""JAX primitive around ``cufinufft_execute`` on a :class:`PlanSet`.

This is the only module that touches JAX internals (``jax._src``): the
primitive needs a transpose rule for the adjoint, which ``jax.ffi.ffi_call``
does not offer. When a JAX bump breaks the backend, the breakage is here.

One primitive, ``cufinufft_exec_p``, covers both transform types.  It always
works on a stack of ``n_trans`` transforms:

- type 2: source ``(n_trans, n_x, n_y)`` on the grid, output ``(n_trans, M)``
  on the points,
- type 1: source ``(n_trans, M)`` on the points, output ``(n_trans, n_x, n_y)``.

The transform is linear in the source, so autodiff needs only a transpose
rule: type 2 transposes to type 1 with the same ``iflag`` and back, on the
same ``PlanSet`` (same convention as jax-finufft).  ``vmap`` over the source
folds the batch into ``n_trans``, which cufinufft runs as one execute.  The
points are never traced; batching over them raises.
"""

from __future__ import annotations

from functools import partial

import numpy as np

import jax
from jax import core
from jax._src import dispatch
from jax._src.interpreters import ad, batching, mlir
from jax.extend.core import Primitive

from ._plan import PlanSet

cufinufft_exec_p = Primitive("cufinufft_exec")


def _abstract_eval(source, *, plan_set: PlanSet, nufft_type: int, iflag: int):
    """Output shape and dtype of one execute, with the input checked.

    JAX calls this while tracing, before any GPU work. Mismatched dtype or
    shape is caught here with a readable message, since the C++ handler
    trusts the buffers it receives to fit the plan.
    """
    if source.dtype != plan_set.dtype:
        raise TypeError(
            f"cufinufft_exec: source dtype {source.dtype} does not match the plan's {plan_set.dtype}"
        )
    if not source.shape or source.shape[0] < 1:
        raise ValueError("cufinufft_exec needs a nonempty leading transform dimension")
    n_trans = source.shape[0]
    if nufft_type == 2:
        expected = (n_trans,) + plan_set.n_modes
        out_shape = (n_trans, plan_set.n_points)
    elif nufft_type == 1:
        expected = (n_trans, plan_set.n_points)
        out_shape = (n_trans,) + plan_set.n_modes
    else:
        raise ValueError(f"nufft_type must be 1 or 2, got {nufft_type}")
    if tuple(source.shape) != expected:
        raise ValueError(
            f"cufinufft_exec type {nufft_type}: source shape {source.shape}, expected {expected}"
        )
    return core.ShapedArray(out_shape, source.dtype)


cufinufft_exec_p.def_abstract_eval(_abstract_eval)
cufinufft_exec_p.def_impl(partial(dispatch.apply_primitive, cufinufft_exec_p))


def _lowering(ctx, source, *, plan_set: PlanSet, nufft_type: int, iflag: int):
    """Emit the FFI call for one execute into the compiled program.

    This is where the plan for ``(nufft_type, iflag, n_trans)`` is built, once
    per compiled shape, and where the executable is made to keep the PlanSet
    alive. The program itself carries only the address of the plan's C++
    handle, see :class:`Plan`.
    """
    plan = plan_set.plan(nufft_type, iflag, ctx.avals_in[0].shape[0])
    # The HLO carries a raw address, which keeps no Python owner alive. Retain
    # the PlanSet (points, plans, stream) even if the tracing closure dies.
    ctx.module_context.add_keepalive(plan_set)
    rule = jax.ffi.ffi_lowering(plan_set._backend.exec.HANDLER_NAME)
    return rule(ctx, source, handle_ptr=np.int64(plan.address))


mlir.register_lowering(cufinufft_exec_p, _lowering, platform="cuda")


def _transpose(ct, source, *, plan_set, nufft_type, iflag):
    """Transpose rule: swap type 1 and type 2, keep ``iflag`` and PlanSet.

    JAX's transpose of a complex linear map is bilinear, not the conjugate
    adjoint, so the exponent sign stays the same (as in jax-finufft). Reverse
    mode gradients of :func:`nufft2` therefore run a type 1 plan on the same
    points, built on first use.
    """
    assert ad.is_undefined_primal(source)
    if type(ct) is ad.Zero:
        return (ad.Zero(source.aval),)
    other = 1 if nufft_type == 2 else 2
    return (
        cufinufft_exec_p.bind(
            ct, plan_set=plan_set, nufft_type=other, iflag=iflag
        ),
    )


ad.deflinear2(cufinufft_exec_p, _transpose)


def _batch(args, dims, *, plan_set, nufft_type, iflag):
    """Batching rule: fold the ``vmap`` axis into ``n_trans``.

    A ``vmap`` over ``B`` sources becomes one execute with ``B * n_trans``
    transforms, which cufinufft runs on the same sorted points in one call.
    This needs its own plan, since ``n_trans`` is fixed when a plan is made.
    """
    (source,), (bdim,) = args, dims
    source = batching.moveaxis(source, bdim, 0)
    batch, n_trans = source.shape[:2]
    flat = source.reshape((batch * n_trans,) + source.shape[2:])
    out = cufinufft_exec_p.bind(
        flat, plan_set=plan_set, nufft_type=nufft_type, iflag=iflag
    )
    return out.reshape((batch, n_trans) + out.shape[1:]), 0


batching.primitive_batchers[cufinufft_exec_p] = _batch


def nufft2(source, plan_set: PlanSet, *, iflag: int = -1):
    """Type 2 NUFFT from the uniform grid ``(n_x, n_y)`` to the ``M`` points.

    Evaluates the Fourier series with coefficients ``source`` at the
    non-uniform points of ``plan_set``; for the radio response this maps a sky
    image to visibilities. Same convention as
    ``jax_finufft.nufft2(source, x, y)``: ``x`` runs along axis 0 of
    ``source`` and ``iflag=-1`` puts a minus sign in the exponent. Works under
    ``jit``, ``grad`` and ``vmap``.

    Parameters
    ----------
    source : array, shape ``plan_set.n_modes``
        Values on the uniform grid, cast to ``plan_set.dtype``.
    plan_set : PlanSet
        Holds the points and the plans. Its plans are built when the calling
        program is compiled.
    iflag : int, optional
        Sign of the exponent, -1 (default) or +1.

    Returns
    -------
    array, shape ``(M,)``
        Values at the points, ``M = plan_set.n_points``.
    """
    source = source.astype(plan_set.dtype)
    out = cufinufft_exec_p.bind(
        source[None], plan_set=plan_set, nufft_type=2, iflag=iflag
    )
    return out[0]


def nufft1(strengths, plan_set: PlanSet, *, iflag: int = 1):
    """Type 1 NUFFT from the ``M`` points to the uniform grid ``(n_x, n_y)``.

    Spreads the values at the non-uniform points of ``plan_set`` onto the
    Fourier grid; for the radio response this maps visibilities to a dirty
    image. With ``iflag`` opposite to that of :func:`nufft2` it is the adjoint
    of :func:`nufft2`. Works under ``jit``, ``grad`` and ``vmap``.

    Parameters
    ----------
    strengths : array, shape ``(M,)``
        Values at the points, cast to ``plan_set.dtype``.
    plan_set : PlanSet
        Holds the points and the plans.
    iflag : int, optional
        Sign of the exponent, +1 (default) or -1.

    Returns
    -------
    array, shape ``plan_set.n_modes``
        Values on the uniform grid.
    """
    strengths = strengths.astype(plan_set.dtype)
    out = cufinufft_exec_p.bind(
        strengths[None], plan_set=plan_set, nufft_type=1, iflag=iflag
    )
    return out[0]
