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
    """Directory of the ``nvidia`` namespace package, or None if absent.

    In a ``jax[cuda12]`` environment the CUDA libraries come as ``nvidia-*``
    wheels below this directory rather than from a system CUDA install.
    """
    try:
        import nvidia
    except ImportError:
        return None
    return Path(nvidia.__file__).parent


def _preload_cuda_runtime():
    """Load the CUDA runtime and cuFFT from the nvidia wheels before cufinufft.

    The PyPI ``libcufinufft.so`` is linked against ``libcudart.so.12`` and
    ``libcufft.so.11`` but does not record where to find them, and in a
    ``jax[cuda12]`` venv they live in the ``nvidia-*`` wheels. Loading them
    with ``RTLD_GLOBAL`` first puts them in the process, so the dynamic linker
    finds them when cufinufft is imported. Missing wheels or libraries are
    skipped silently; the cufinufft import then reports the real failure.
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
    """Import the low level ctypes bindings of libcufinufft.

    We use ``cufinufft._cufinufft`` rather than the high level
    ``cufinufft.Plan`` because we need the raw plan pointer to hand to the C++
    handler, and full control over when the plan is created and destroyed.
    The CUDA libraries are preloaded first, see :func:`_preload_cuda_runtime`.
    """
    _preload_cuda_runtime()
    import cufinufft._cufinufft as bindings

    return bindings


class CudaRuntime:
    """The few CUDA runtime calls the backend makes from Python.

    A CUDA stream is an ordered queue of GPU work: operations submitted to one
    stream run in order, while work on different streams may overlap. JAX runs
    its programs on its own stream; :class:`PlanSet` creates a second one for
    cufinufft, because a cufinufft plan is bound to one stream when it is made.
    This class creates, waits on and destroys such streams through libcudart,
    and hands the C++ handler the addresses of the event functions it uses to
    order work between the two streams.

    Streams are passed around as plain integers holding the CUDA pointer.
    """

    def __init__(self):
        """Load libcudart and declare the signatures of the calls we use.

        Raises
        ------
        ImportError
            If no ``libcudart.so.12`` can be loaded.
        """
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
        """Open libcudart, preferring the copy from the nvidia wheels.

        The wheel copy is tried first so that we share the runtime that JAX
        and cufinufft already use; then whatever ``find_library`` reports,
        then the bare SONAME.
        """
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
        """Memory address of the libcudart function ``name``.

        The C++ handler is compiled without CUDA headers, so it receives the
        CUDA functions it calls as raw function addresses from here.

        Parameters
        ----------
        name : str
            Symbol name in libcudart, e.g. ``"cudaEventRecord"``.

        Returns
        -------
        int
            The function address.
        """
        return ctypes.cast(getattr(self.lib, name), ctypes.c_void_p).value

    def stream_create(self) -> int:
        """Create a new non-blocking CUDA stream.

        Non-blocking means the stream does not implicitly wait for the legacy
        default stream, so it only orders against the streams we tell it to.

        Returns
        -------
        int
            The stream pointer.

        Raises
        ------
        RuntimeError
            If CUDA reports an error.
        """
        stream = ctypes.c_void_p()
        non_blocking = 0x01
        ret = self.lib.cudaStreamCreateWithFlags(ctypes.byref(stream), non_blocking)
        if ret != 0:
            raise RuntimeError(f"cudaStreamCreateWithFlags failed with CUDA error {ret}")
        return stream.value

    def stream_destroy(self, stream: int) -> None:
        """Destroy ``stream``. Work already queued on it still completes."""
        self.lib.cudaStreamDestroy(ctypes.c_void_p(stream))

    def stream_synchronize(self, stream: int) -> None:
        """Block the calling thread until all work queued on ``stream`` is done.

        Raises
        ------
        RuntimeError
            If CUDA reports an error, which may come from any earlier work on
            the stream.
        """
        ret = self.lib.cudaStreamSynchronize(ctypes.c_void_p(stream))
        if ret != 0:
            raise RuntimeError(f"cudaStreamSynchronize failed with CUDA error {ret}")


def _import_exec_handler(cuda: CudaRuntime) -> ModuleType:
    """Import the compiled FFI handler, check it, and register it with JAX.

    An XLA FFI handler is a C++ function that XLA calls from inside a compiled
    program, passing it the input and output device buffers and the stream the
    program runs on. Ours lives in ``_exec.cpp`` and runs ``cufinufft_execute``
    on a plan made in Python. Before registration it is told the addresses of
    the CUDA event functions, which it uses to make the cufinufft stream wait
    for JAX's stream and back.

    Parameters
    ----------
    cuda : CudaRuntime
        Supplies the CUDA event function addresses.

    Returns
    -------
    ModuleType
        The ``_exec`` extension module.

    Raises
    ------
    ImportError
        If ``_exec`` was not built, or was built against a different jaxlib.
        XLA would otherwise drop the handler at registration without an error.
    """
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
    """The loaded native pieces, as returned by :func:`load`.

    Parameters
    ----------
    cuda : CudaRuntime
        Stream management and CUDA function addresses.
    cuf : ModuleType
        ``cufinufft._cufinufft``, the ctypes bindings of libcufinufft with
        argument types declared (makeplan, setpts, execute, destroy, opts).
    exec : ModuleType
        The compiled ``_exec`` module holding the FFI handler, already
        registered with JAX.
    """

    cuda: CudaRuntime
    cuf: ModuleType
    exec: ModuleType


_lock = threading.Lock()
_backend: Union[Backend, None] = None


def load() -> Backend:
    """Load the native libraries once per process and return them.

    Later calls return the same :class:`Backend`. The lock makes the first
    load safe when several threads construct a :class:`PlanSet` at once.

    Returns
    -------
    Backend

    Raises
    ------
    ImportError
        If cufinufft, libcudart or the compiled handler cannot be loaded.
    """
    global _backend
    with _lock:
        if _backend is None:
            cuf = _import_cufinufft_bindings()
            cuda = CudaRuntime()
            _backend = Backend(cuda=cuda, cuf=cuf, exec=_import_exec_handler(cuda))
        return _backend
