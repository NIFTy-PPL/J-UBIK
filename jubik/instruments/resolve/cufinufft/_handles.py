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

A ``PlanSet`` owns the non-uniform points of one response (as committed JAX
arrays on the CUDA device), one CUDA stream, and a cache of ``ExecutablePlan``
objects indexed by ``PlanKey(nufft_type, iflag, n_trans)``.
Construction uploads points; lowering lazily creates each plan and sorts
its points once. Compiled calls then only execute (see ``_exec.cpp``).

Each executable retains the ``PlanSet`` because its native handle addresses
are embedded as custom-call attributes. Resources are released only after
the last owner disappears; there is deliberately no explicit close method.
"""

from __future__ import annotations

import ctypes
import glob
import threading
from ctypes.util import find_library
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

import jax
import jax.numpy as jnp

_NVIDIA_LIBS = (
    "cuda_runtime/lib/libcudart.so.12",
    "cufft/lib/libcufft.so.11",
)


def _nvidia_wheel_root():
    try:
        import nvidia
    except ImportError:
        return None
    return Path(nvidia.__file__).parent


def _preload_cuda_runtime():
    """Load the CUDA runtime and cuFFT by SONAME before cufinufft.

    The PyPI ``libcufinufft.so`` is linked against ``libcudart.so.12`` and
    ``libcufft.so.11`` but carries no rpath; in a ``jax[cuda12]`` venv they
    live in the ``nvidia-*`` wheels.  Loading them ``RTLD_GLOBAL`` first lets
    the dynamic linker resolve them when cufinufft is imported.
    """
    root = _nvidia_wheel_root()
    if root is None:
        return
    for pattern in _NVIDIA_LIBS:
        for hit in glob.glob(str(root / pattern)):
            try:
                ctypes.CDLL(hit, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
            break


def import_cufinufft():
    _preload_cuda_runtime()
    import cufinufft

    return cufinufft


class _CudaRuntime:
    """The handful of CUDA runtime calls the plan machinery needs."""

    def __init__(self):
        self.lib = self._load()
        self.lib.cudaStreamCreateWithFlags.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
        ]
        self.lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        for fn in (
            self.lib.cudaStreamCreateWithFlags,
            self.lib.cudaStreamDestroy,
            self.lib.cudaStreamSynchronize,
        ):
            fn.restype = ctypes.c_int

    @staticmethod
    def _load():
        root = _nvidia_wheel_root()
        candidates = []
        if root is not None:
            candidates += glob.glob(str(root / _NVIDIA_LIBS[0]))
        found = find_library("cudart")
        if found:
            candidates.append(found)
        candidates.append("libcudart.so.12")
        for cand in candidates:
            try:
                return ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
        raise ImportError("libcudart.so.12 not found; the cufinufft backend needs the CUDA runtime")

    def address(self, name: str) -> int:
        return ctypes.cast(getattr(self.lib, name), ctypes.c_void_p).value

    def stream_create(self) -> int:
        stream = ctypes.c_void_p()
        non_blocking = 0x01
        ret = self.lib.cudaStreamCreateWithFlags(ctypes.byref(stream), non_blocking)
        if ret != 0:
            raise RuntimeError(f"cudaStreamCreateWithFlags failed with CUDA error {ret}")
        return stream.value

    def stream_destroy(self, stream: int) -> None:
        self.lib.cudaStreamDestroy(ctypes.c_void_p(stream))

    def stream_synchronize(self, stream: int) -> None:
        ret = self.lib.cudaStreamSynchronize(ctypes.c_void_p(stream))
        if ret != 0:
            raise RuntimeError(f"cudaStreamSynchronize failed with CUDA error {ret}")


_lock = threading.Lock()
_runtime: Union[_CudaRuntime, None] = None
_exec_lib = None
_cufinufft = None


def runtime() -> _CudaRuntime:
    """The CUDA runtime, the cufinufft module and the FFI handler, loaded once.

    Order matters: cufinufft first (it pins libcudart into the process), then
    the handler module, which receives the addresses of ``cufinufft_execute``
    and the CUDA event/stream calls and is registered with JAX under
    ``"cufinufft_exec"`` for the CUDA platform.
    """
    global _runtime, _exec_lib, _cufinufft
    with _lock:
        if _runtime is not None:
            return _runtime
        _cufinufft = import_cufinufft()
        rt = _CudaRuntime()
        cuf = _cufinufft._cufinufft.lib
        try:
            from . import _exec
        except ImportError as err:
            raise ImportError(
                "cufinufft FFI handler not built (jubik.instruments.resolve.cufinufft._exec); "
                "reinstall jubik with a C++ compiler available."
            ) from err
        import jaxlib

        if getattr(_exec, "JAXLIB_VERSION", None) != jaxlib.__version__:
            raise ImportError(
                "cufinufft FFI handler was compiled against jaxlib "
                f"{getattr(_exec, 'JAXLIB_VERSION', 'unknown')} but jaxlib "
                f"{jaxlib.__version__} is installed; "
                "XLA drops handlers with a mismatched FFI API version at registration. "
                "Rebuild jubik without build isolation (see setup.py)."
            )
        _exec.init(
            ctypes.cast(cuf.cufinufft_execute, ctypes.c_void_p).value,
            ctypes.cast(cuf.cufinufftf_execute, ctypes.c_void_p).value,
            rt.address("cudaEventCreateWithFlags"),
            rt.address("cudaEventRecord"),
            rt.address("cudaStreamWaitEvent"),
            rt.address("cudaEventDestroy"),
        )
        for name, capsule in _exec.registrations().items():
            jax.ffi.register_ffi_target(name, capsule, platform="CUDA")
        _exec_lib = _exec
        _runtime = rt
        return rt


class _Flags:
    c_contiguous = True


class _DeviceView:
    """What ``cufinufft.Plan.setpts`` wants to see of a committed JAX array.

    The cufinufft Python layer reads ``__cuda_array_interface__`` for the
    pointer and a few numpy-like attributes for its checks; JAX arrays carry
    the former but not ``.flags``.  A contiguous device array is exactly what
    ``jnp.asarray`` hands back, so this view is honest.
    """

    def __init__(self, arr):
        self._arr = arr
        self.__cuda_array_interface__ = arr.__cuda_array_interface__
        self.shape = arr.shape
        self.ndim = arr.ndim
        self.size = arr.size
        self.dtype = arr.dtype
        self.flags = _Flags()


@dataclass(frozen=True, order=True)
class PlanKey:
    """Transform settings that require a distinct native cuFINUFFT plan."""

    nufft_type: int
    iflag: int
    n_trans: int


@dataclass
class ExecutablePlan:
    """Own one cuFINUFFT plan and its C++ PlanHandle capsule.

    ``address`` points to the capsule-owned handle passed to the compiled FFI
    call. The parent PlanSet keeps the coordinate arrays and stream alive.
    """

    plan: object
    handle: object
    address: int


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
        (``min(n_trans, 8)`` grids per plan).  1 keeps a plan at one grid of
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
        self._plans: dict[PlanKey, ExecutablePlan] = {}
        self._plan_lock = threading.Lock()
        self.stream = None
        self.dtype = np.dtype(dtype)
        if self.dtype == np.complex128:
            real = jnp.float64
        elif self.dtype == np.complex64:
            real = jnp.float32
        else:
            raise TypeError(f"dtype must be complex64 or complex128, got {dtype}")
        if self.dtype == np.complex128 and not jax.config.jax_enable_x64:
            raise RuntimeError("complex128 plans need jax_enable_x64")

        self.n_modes = tuple(int(n) for n in n_modes)
        if len(self.n_modes) != 2:
            raise NotImplementedError("PlanSet supports 2D transforms only")
        self.eps = float(eps)
        self.gpu_maxbatchsize = int(gpu_maxbatchsize)
        self.upsampfac = float(upsampfac)

        self.x = jnp.asarray(x, dtype=real)
        self.y = jnp.asarray(y, dtype=real)
        if self.x.ndim != 1 or self.x.shape != self.y.shape:
            raise ValueError("x and y must be 1D of equal length")
        (device,) = self.x.devices()
        if device.platform != "gpu":
            raise RuntimeError(f"PlanSet needs the points on a CUDA device, got {device}")
        self.device_id = device.id
        self.n_points = int(self.x.shape[0])

        self._rt = runtime()
        self.stream = self._rt.stream_create()

    def _handle(self, nufft_type: int, iflag: int, n_trans: int) -> int:
        """Native handle for this transform, built once even under concurrent lowering."""
        key = PlanKey(int(nufft_type), int(iflag), int(n_trans))
        # This lock protects lazy construction. Execution is protected by the
        # separate native mutex stored in each handle, without the Python GIL.
        with self._plan_lock:
            if key not in self._plans:
                plan = _cufinufft.Plan(
                    key.nufft_type,
                    self.n_modes,
                    n_trans=key.n_trans,
                    eps=self.eps,
                    isign=key.iflag,
                    dtype=self.dtype,
                    gpu_stream=self.stream,
                    gpu_device_id=self.device_id,
                    gpu_maxbatchsize=self.gpu_maxbatchsize,
                    upsampfac=self.upsampfac,
                )
                plan.setpts(_DeviceView(self.x), _DeviceView(self.y))
                self._rt.stream_synchronize(self.stream)
                handle = _exec_lib.make_plan_handle(plan._plan.value, self.stream)
                self._plans[key] = ExecutablePlan(
                    plan=plan,
                    handle=handle,
                    address=_exec_lib.plan_handle_address(handle),
                )
            return self._plans[key].address

    @property
    def n_plans(self) -> int:
        return len(self._plans)

    def __del__(self):
        try:
            if self.stream is not None:
                # The last executable can disappear while its GPU work is
                # still queued. Finish that work before releasing resources.
                self._rt.stream_synchronize(self.stream)
                for executable in self._plans.values():
                    executable.plan.__del__()
                self._plans.clear()
                self._rt.stream_destroy(self.stream)
                self.stream = None
        except Exception:
            pass

    def __repr__(self):
        return (
            f"PlanSet(n_modes={self.n_modes}, n_points={self.n_points}, eps={self.eps}, "
            f"dtype={self.dtype}, plans={sorted(self._plans)})"
        )
