"""Build hook for the one compiled piece of jubik.

Everything else is pure Python and configured in pyproject.toml.  The
cufinufft FFI handler (``jubik/instruments/resolve/cufinufft/_exec.cpp``)
needs a C++17 compiler and the XLA FFI headers that jaxlib ships; without them
the extension is skipped and the ``cufinufft`` radio response backend raises
at construction.

The headers must come from the jaxlib the handler will run against: a handler
compiled against a newer XLA FFI API than the installed CUDA plugin is
silently dropped at registration and every call fails with ``NOT_FOUND``.
Build without isolation (``uv sync`` with ``no-build-isolation-package =
["jubik"]``, or ``pip install --no-build-isolation``) so ``import jax`` here
resolves to the runtime environment, or point ``XLA_FFI_INCLUDE_DIR`` at the
right ``jaxlib/include``. The jaxlib version seen at build time is embedded
in the extension and checked at import.
"""

import os
from pathlib import Path

from setuptools import Extension, setup

PACKAGE_DIR = Path("jubik/instruments/resolve/cufinufft")


def _xla_headers():
    """(include dir, jaxlib version) or (None, None)."""
    override = os.environ.get("XLA_FFI_INCLUDE_DIR")
    try:
        import jaxlib

        version = jaxlib.__version__
    except Exception:
        version = None
    if override:
        return override, version
    try:
        import jax.ffi

        return jax.ffi.include_dir(), version
    except Exception:
        return None, version


INCLUDE_DIR, JAXLIB_VERSION = _xla_headers()


cufinufft_exec = Extension(
    "jubik.instruments.resolve.cufinufft._exec",
    sources=[str(PACKAGE_DIR / "_exec.cpp")],
    include_dirs=[INCLUDE_DIR] if INCLUDE_DIR else [],
    define_macros=[("JUBIK_JAXLIB_VERSION", f'"{JAXLIB_VERSION}"')],
    language="c++",
    extra_compile_args=["-std=c++17", "-O2"],
    optional=True,
)

setup(ext_modules=[cufinufft_exec])
