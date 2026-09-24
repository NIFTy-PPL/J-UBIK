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

"""cufinufft plans that outlive a single call.

The cost of one NUFFT call is ``P + S(N) + k [G + I]``: the plan ``P``
(cuFFT plan, kernel spectrum, work arrays, about 8 ms) and the point sort
``S`` are paid on every ``jax_finufft`` call, and for our data they dominate.
A :class:`PlanSet` pays them once. It owns the non-uniform points of one
response as JAX arrays on the CUDA device, one CUDA stream, and one
:class:`Plan` per ``(nufft_type, iflag, n_trans)`` that a compiled program
asked for. Compiled programs then only execute (see ``_exec.cpp``).

Ownership, top down: PlanSet owns stream and plans; each Plan owns its
native cufinufft plan and the C++ handle the compiled program addresses;
the handle borrows the stream. Executables retain their PlanSet, so
``close()`` is optional; ``__del__`` calls it when the last owner is gone.
"""

from __future__ import annotations

import ctypes
import threading

import numpy as np

import jax
import jax.numpy as jnp

from ._libs import Backend, load

_REAL = {np.dtype(np.complex128): np.float64, np.dtype(np.complex64): np.float32}


def _device_pointer(arr) -> int:
    return arr.__cuda_array_interface__["data"][0]


class Plan:
    """One native cufinufft plan with its points set and sorted.

    ``address`` is what the compiled program carries: the C++ handle that
    bundles the plan, its stream, the execute function and the transform type.
    """

    def __init__(self, backend: Backend, plan_ptr: int, handle, *, nufft_type: int, iflag: int, n_trans: int, dtype):
        self._backend = backend
        self._plan = plan_ptr
        self._handle = handle
        self.nufft_type = nufft_type
        self.iflag = iflag
        self.n_trans = n_trans
        self.dtype = dtype
        self.address = backend.exec.plan_handle_address(handle)

    def close(self) -> None:
        if self._plan is None:
            return
        destroy = self._backend.cuf._destroy_plan if self.dtype == np.complex128 else self._backend.cuf._destroy_planf
        destroy(ctypes.c_void_p(self._plan))
        self._plan = None
        self._handle = None

    def __repr__(self):
        return f"Plan(type={self.nufft_type}, iflag={self.iflag}, n_trans={self.n_trans})"


class PlanSet:
    """The cufinufft plans of one set of non-uniform points.

    Parameters
    ----------
    n_modes : tuple of int
        Grid shape ``(n_x, n_y)`` of the uniform side, in the order of the
        array axes fed to :func:`nufft2` (``x`` runs along axis 0).
    x, y : array-like
        Non-uniform coordinates in radians, ``[-pi, pi)`` or ``[0, 2 pi)``.
    eps : float
        Requested precision, as in jax-finufft.
    dtype : complex dtype
        ``complex128`` (points ``float64``) or ``complex64``.
    gpu_maxbatchsize : int
        cufinufft's ``gpu_maxbatchsize``; 0 leaves its heuristic
        (``min(n_trans, 8)`` grids per plan). 1 keeps a plan at one grid of
        memory for any ``n_trans``.
    upsampfac : float
        Oversampling factor sigma of the fine grid.
    """

    def __init__(
        self,
        n_modes,
        x,
        y,
        *,
        eps: float,
        dtype=np.complex128,
        gpu_maxbatchsize: int = 0,
        upsampfac: float = 2.0,
    ):
        self._plans: dict[tuple[int, int, int], Plan] = {}
        self._lock = threading.Lock()
        self.stream = None

        self.dtype = np.dtype(dtype)
        if self.dtype not in _REAL:
            raise TypeError(f"dtype must be complex64 or complex128, got {dtype}")
        if self.dtype == np.complex128 and not jax.config.jax_enable_x64:
            raise RuntimeError("complex128 plans need jax_enable_x64")
        self.n_modes = tuple(int(n) for n in n_modes)
        if len(self.n_modes) != 2:
            raise NotImplementedError("PlanSet supports 2D transforms only")
        self.eps = float(eps)
        self.gpu_maxbatchsize = int(gpu_maxbatchsize)
        self.upsampfac = float(upsampfac)

        self.x = jnp.asarray(x, dtype=_REAL[self.dtype])
        self.y = jnp.asarray(y, dtype=_REAL[self.dtype])
        if self.x.ndim != 1 or self.x.shape != self.y.shape:
            raise ValueError("x and y must be 1D of equal length")
        (device,) = self.x.devices()
        if device.platform != "gpu":
            raise RuntimeError(f"PlanSet needs the points on a CUDA device, got {device}")
        self.device_id = device.id
        self.n_points = int(self.x.shape[0])

        self._backend = load()
        self.stream = self._backend.cuda.stream_create()

    def plan(self, nufft_type: int, iflag: int, n_trans: int) -> Plan:
        """The plan for this transform, built on first request."""
        key = (int(nufft_type), int(iflag), int(n_trans))
        # This lock protects lazy construction. Execution is protected by the
        # native mutex in each handle, without the Python GIL.
        with self._lock:
            if key not in self._plans:
                self._plans[key] = self._build(*key)
            return self._plans[key]

    def _build(self, nufft_type: int, iflag: int, n_trans: int) -> Plan:
        if self.stream is None:
            raise RuntimeError("PlanSet is closed")
        cuf = self._backend.cuf
        double = self.dtype == np.complex128
        make_plan = cuf._make_plan if double else cuf._make_planf
        set_pts = cuf._set_pts if double else cuf._set_ptsf
        execute = cuf._exec_plan if double else cuf._exec_planf

        opts = cuf.NufftOpts()
        cuf._default_opts(opts)
        opts.gpu_stream = self.stream
        opts.gpu_device_id = self.device_id
        opts.gpu_maxbatchsize = self.gpu_maxbatchsize
        opts.upsampfac = self.upsampfac

        # libcufinufft is column major: modes are passed as (n_x, n_y, n_z)
        # reversed from the array axes, and the point axes swap accordingly.
        # This mirrors cufinufft.Plan and keeps x along axis 0 of the grid.
        n_x, n_y = self.n_modes
        modes = (ctypes.c_int64 * 3)(n_y, n_x, 1)
        plan = ctypes.c_void_p()
        ier = make_plan(nufft_type, 2, modes, iflag, n_trans, self.eps, ctypes.byref(plan), ctypes.byref(opts))
        if ier != 0:
            raise RuntimeError(f"cufinufft_makeplan failed with code {ier}")
        try:
            ier = set_pts(
                plan, self.n_points, _device_pointer(self.y), _device_pointer(self.x), None,
                0, None, None, None,
            )
            if ier != 0:
                raise RuntimeError(f"cufinufft_setpts failed with code {ier}")
            self._backend.cuda.stream_synchronize(self.stream)
        except Exception:
            (cuf._destroy_plan if double else cuf._destroy_planf)(plan)
            raise
        handle = self._backend.exec.make_plan_handle(
            plan.value, self.stream, ctypes.cast(execute, ctypes.c_void_p).value, nufft_type
        )
        return Plan(
            self._backend, plan.value, handle,
            nufft_type=nufft_type, iflag=iflag, n_trans=n_trans, dtype=self.dtype,
        )

    @property
    def n_plans(self) -> int:
        return len(self._plans)

    def close(self) -> None:
        """Release plans and stream. Idempotent.

        Order matters: queued work on the stream must finish before the plans
        it uses are destroyed, and the plans before the stream they were
        created on.
        """
        with self._lock:
            if self.stream is None:
                return
            self._backend.cuda.stream_synchronize(self.stream)
            for plan in self._plans.values():
                plan.close()
            self._plans.clear()
            self._backend.cuda.stream_destroy(self.stream)
            self.stream = None

    def __del__(self):
        if getattr(self, "stream", None) is None:
            return
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return (
            f"PlanSet(n_modes={self.n_modes}, n_points={self.n_points}, eps={self.eps}, "
            f"dtype={self.dtype}, plans={sorted(self._plans)})"
        )
