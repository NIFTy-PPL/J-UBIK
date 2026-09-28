"""Locating the CUDA libraries of the nvidia wheels, no GPU needed."""

import importlib
import sys

from jubik.instruments.resolve.cufinufft import _libs


def test_nvidia_namespace_package(tmp_path, monkeypatch):
    # Most nvidia-* wheels ship no nvidia/__init__.py, so `nvidia` is a
    # namespace package with __file__ None, possibly split over several
    # site-packages directories.
    first, second = tmp_path / "site_a", tmp_path / "site_b"
    (first / "nvidia" / "cufft" / "lib").mkdir(parents=True)
    (second / "nvidia" / "cuda_runtime" / "lib").mkdir(parents=True)
    cudart = second / "nvidia" / "cuda_runtime" / "lib" / "libcudart.so.12"
    cudart.write_bytes(b"")

    monkeypatch.setattr(sys, "path", [str(first), str(second)])
    monkeypatch.delitem(sys.modules, "nvidia", raising=False)
    importlib.invalidate_caches()
    try:
        assert _libs._nvidia_wheel_roots() == [first / "nvidia", second / "nvidia"]
        assert _libs._nvidia_wheel_libs(_libs._NVIDIA_LIBS[0]) == [str(cudart)]
        # The empty stand-in fails to load and is skipped.
        _libs._preload_cuda_runtime()
    finally:
        sys.modules.pop("nvidia", None)
