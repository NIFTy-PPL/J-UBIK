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

"""The three native pieces the cufinufft backend runs on, loaded once.

``load()`` returns a :class:`Backend` holding

- ``cuda``: the few CUDA runtime calls we make from Python (streams),
- ``cuf``: ``cufinufft._cufinufft``, the ctypes bindings of libcufinufft with
  argtypes already declared (makeplan, setpts, execute, destroy, opts),
- ``exec``: the compiled ``_exec`` module, whose XLA FFI handler runs
  ``cufinufft_execute`` inside compiled JAX programs.

Nothing here knows about plans or points. Loading order matters: cufinufft
first (it pins ``libcudart`` into the process), then the handler, which is
told the CUDA event and stream-wait entry points and registered with JAX.
"""

from __future__ import annotations

import ctypes
import glob
import threading
from ctypes.util import find_library
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Union

import jax

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
    live in the ``nvidia-*`` wheels. Loading them ``RTLD_GLOBAL`` first lets
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


def _import_cufinufft_bindings() -> ModuleType:
    _preload_cuda_runtime()
    import cufinufft._cufinufft as bindings

    return bindings


class CudaRuntime:
    """Stream creation, destruction and synchronisation through libcudart."""

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


def _import_exec_handler(cuda: CudaRuntime) -> ModuleType:
    """Import, check and register the compiled FFI handler."""
    try:
        from . import _exec
    except ImportError as err:
        raise ImportError(
            "cufinufft FFI handler not built (jubik.instruments.resolve.cufinufft._exec); "
            "reinstall jubik with a C++ compiler available."
        ) from err
    import jaxlib

    built_against = getattr(_exec, "JAXLIB_VERSION", "unknown")
    if built_against != jaxlib.__version__:
        raise ImportError(
            f"cufinufft FFI handler was compiled against jaxlib {built_against} but jaxlib "
            f"{jaxlib.__version__} is installed; XLA drops handlers with a mismatched FFI "
            "API version at registration. Rebuild jubik without build isolation (see setup.py)."
        )
    _exec.init(
        cuda.address("cudaEventCreateWithFlags"),
        cuda.address("cudaEventRecord"),
        cuda.address("cudaStreamWaitEvent"),
        cuda.address("cudaEventDestroy"),
    )
    jax.ffi.register_ffi_target(_exec.HANDLER_NAME, _exec.handler(), platform="CUDA")
    return _exec


@dataclass(frozen=True)
class Backend:
    cuda: CudaRuntime
    cuf: ModuleType
    exec: ModuleType


_lock = threading.Lock()
_backend: Union[Backend, None] = None


def load() -> Backend:
    global _backend
    with _lock:
        if _backend is None:
            cuf = _import_cufinufft_bindings()
            cuda = CudaRuntime()
            _backend = Backend(cuda=cuda, cuf=cuf, exec=_import_exec_handler(cuda))
        return _backend
