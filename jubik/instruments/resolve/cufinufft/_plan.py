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
    """GPU memory address of a JAX array's buffer.

    libcufinufft takes the non-uniform points as raw device pointers. The
    caller must keep ``arr`` alive for as long as the pointer is used; here
    :class:`PlanSet` holds the point arrays for the lifetime of its plans.
    """
    return arr.__cuda_array_interface__["data"][0]


class Plan:
    """One native cufinufft plan with its points set and sorted.

    A cufinufft plan is a prepared transform: it fixes the grid shape, the
    transform type (1 is points to grid, 2 is grid to points), the sign of the
    exponent, the precision, the requested accuracy and the number ``n_trans``
    of transforms run together. Making it builds the cuFFT plan, the kernel's
    Fourier coefficients and the work arrays; setting the points then sorts
    them into bins. After that the plan can execute repeatedly with new data
    at the same points, which is the only step a compiled program performs.

    Instances are made by :meth:`PlanSet.plan`, never directly.

    Parameters
    ----------
    backend : Backend
        Loaded native libraries, used to destroy the plan.
    plan_ptr : int
        Pointer to the native cufinufft plan, owned by this object.
    handle : PyCapsule
        The C++ ``PlanHandle`` that the FFI handler receives. It bundles the
        plan pointer, the stream, the precision-specific execute function and
        the transform type. A PyCapsule is a Python object wrapping a raw C
        pointer together with a destructor, so the handle is freed by Python
        reference counting once this object drops it.
    nufft_type : int
        1 (points to grid) or 2 (grid to points).
    iflag : int
        Sign of the exponent, -1 or +1.
    n_trans : int
        Number of transforms executed together on the same points.
    dtype : numpy dtype
        ``complex64`` or ``complex128``.

    Attributes
    ----------
    address : int
        Address of the C++ handle. This integer is what the compiled program
        carries as an attribute of the FFI call.
    """

    def __init__(self, backend: Backend, plan_ptr: int, handle, *, nufft_type: int, iflag: int, n_trans: int, dtype):
        """Wrap an already built native plan and its handle, see :class:`Plan`."""
        self._backend = backend
        self._plan = plan_ptr
        self._handle = handle
        self.nufft_type = nufft_type
        self.iflag = iflag
        self.n_trans = n_trans
        self.dtype = dtype
        self.address = backend.exec.plan_handle_address(handle)

    def close(self) -> None:
        """Destroy the native plan and drop the handle. Idempotent.

        The caller must make sure no queued GPU work still uses the plan;
        :meth:`PlanSet.close` synchronizes the stream before calling this.
        """
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

    A plan's key ``(nufft_type, iflag, n_trans)`` is fixed when it is made, and
    one response needs several: the forward sky to visibility call uses
    ``(2, -1, 1)``, its gradient ``(1, -1, 1)``, and a ``vmap`` over 8 skies
    ``(2, -1, 8)``. These share points, grid, precision and stream, so one
    PlanSet groups them and builds each on first request. It also owns the
    CUDA stream all its plans run on (see :class:`CudaRuntime`).

    Parameters
    ----------
    n_modes : tuple of int
        Grid shape ``(n_x, n_y)`` of the uniform side, in the order of the
        array axes fed to :func:`nufft2` (``x`` runs along axis 0).
    x, y : array-like
        Non-uniform coordinates in radians, ``[-pi, pi)`` or ``[0, 2 pi)``.
    eps : float
        Requested relative accuracy of the transform, as in jax-finufft.
    dtype : complex dtype
        ``complex128`` (points ``float64``, needs ``jax_enable_x64``) or
        ``complex64`` (points ``float32``).
    gpu_maxbatchsize : int
        How many of the ``n_trans`` transforms cufinufft processes together in
        one pass over the fine grid. 0 lets the library choose
        (``min(n_trans, 8)``); 1 keeps the workspace at one fine grid for any
        ``n_trans``, which saves memory at some cost in speed. Not part of the
        plan key.
    upsampfac : float
        Ratio of the internal fine FFT grid to the requested grid, per
        dimension. 2.0 is the default; 1.25 uses a smaller FFT grid, which
        saves plan time (about 2.5 ms) and memory but needs a wider kernel;
        on the ALMA data it does not change execute time.

    Raises
    ------
    TypeError
        If ``dtype`` is not ``complex64`` or ``complex128``.
    RuntimeError
        If ``complex128`` is asked for without ``jax_enable_x64``, or the
        points do not live on a CUDA device.
    NotImplementedError
        If ``n_modes`` is not two dimensional.
    ValueError
        If ``x`` and ``y`` are not 1D arrays of equal length.

    Notes
    -----
    Lifetime: when a program using this PlanSet is compiled, the lowering
    registers the PlanSet as a keepalive of the executable, so every compiled
    executable keeps its plans, points and stream alive. :meth:`close` is
    therefore optional; ``__del__`` calls it once the last owner is gone.

    Threading: ``_lock`` guards lazy plan construction from Python. Concurrent
    executions of the same plan are serialized in C++ by a per-plan mutex,
    which does not need the Python GIL.
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
        """The plan for this transform, built on first request.

        Called during lowering, so the plan is made and its points sorted once
        per compiled program shape, not once per evaluation.

        Parameters
        ----------
        nufft_type : int
            1 (points to grid) or 2 (grid to points).
        iflag : int
            Sign of the exponent, -1 or +1.
        n_trans : int
            Number of transforms executed together.

        Returns
        -------
        Plan
            The cached plan for ``(nufft_type, iflag, n_trans)``.

        Raises
        ------
        RuntimeError
            If the PlanSet is closed or cufinufft fails to make the plan.
        """
        key = (int(nufft_type), int(iflag), int(n_trans))
        # This lock protects lazy construction. Execution is protected by the
        # native mutex in each handle, without the Python GIL.
        with self._lock:
            if key not in self._plans:
                self._plans[key] = self._build(*key)
            return self._plans[key]

    def _build(self, nufft_type: int, iflag: int, n_trans: int) -> Plan:
        """Make a native plan on this set's stream, set its points, wrap it.

        This pays the plan cost and the point sort once. The stream is
        synchronized after setting the points so that a failure surfaces here
        rather than in the first execute, and a failed plan is destroyed before
        the error propagates.
        """
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
        """Number of plans built so far."""
        return len(self._plans)

    def close(self) -> None:
        """Release plans and stream. Idempotent.

        Order matters: first synchronize the stream, because queued work
        references the plans; then destroy the plans, because they reference
        the stream; then destroy the stream. After closing, requesting a new
        plan raises. Calling this is optional, see the Notes of
        :class:`PlanSet`.
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
        """Close on garbage collection, ignoring errors during shutdown."""
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
