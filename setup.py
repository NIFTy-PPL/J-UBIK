"""Build hook for the one compiled piece of jubik.

Everything else is pure Python and configured in pyproject.toml.  The
cufinufft FFI handler (``jubik/instruments/resolve/cufinufft/_exec.cpp``)
needs a C++17 compiler and nothing else: no CUDA toolkit and no jax at build
time.  Without a compiler the extension is skipped and the ``cufinufft``
radio response backend raises at construction with reinstall instructions.

The XLA FFI headers are vendored in ``cufinufft/include`` (see the README
there), so the build works in pip's and uv's isolated build environments.
XLA accepts handlers built against older FFI headers, so the vendored copy
sets the minimum jaxlib, which the ``resolve-cuda`` extra pins.
"""

from pathlib import Path

from setuptools import Extension, setup

PACKAGE_DIR = Path("jubik/instruments/resolve/cufinufft")


cufinufft_exec = Extension(
    "jubik.instruments.resolve.cufinufft._exec",
    sources=[str(PACKAGE_DIR / "_exec.cpp")],
    include_dirs=[str(PACKAGE_DIR / "include")],
    depends=[str(p) for p in sorted((PACKAGE_DIR / "include").rglob("*.h"))],
    language="c++",
    extra_compile_args=["-std=c++17", "-O2"],
    optional=True,
)

setup(ext_modules=[cufinufft_exec])
