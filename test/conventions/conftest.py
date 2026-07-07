"""Shared setup for the canonical sky-frame convention sweep.

This test package is the pytest twin of ``probes/`` (see ``probes/README.md``):
the probes remain the human-facing consult record, these tests give the same
guarantees breadth-first (rectangles, anisotropy, both radio backends, all four
pointing quadrants, adjoint consistency) under ``pytest``.

The canonical frame every test is judged against (``probes/check_frames.py``):

    sky[i, j]:  i = dim 0  ->  +Dec (North)   [row]
                j = dim 1  ->  -RA  (West)    [column]

``JAX_PLATFORMS`` is pinned to ``cpu`` at import time — BEFORE jax is imported
(transitively, via jubik) — because the jaxbind ducc kernels have no GPU FFI
handler in this environment.
"""

import os

# Must run before any (transitive) jax import.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
from pathlib import Path

import pytest

# --- repo-root / probes locations -------------------------------------------
# conftest.py lives at <repo>/test/conventions/conftest.py
REPO_ROOT = Path(__file__).resolve().parents[2]
PROBES_DIR = REPO_ROOT / "probes"
GOLDEN_DIR = PROBES_DIR / "golden"
ROUNDTRIP_DIR = PROBES_DIR / "roundtrip"

# Make the orientation glyph module importable (probes/roundtrip/glyph.py), the
# same way p6/p7 do it.
if str(ROUNDTRIP_DIR) not in sys.path:
    sys.path.insert(0, str(ROUNDTRIP_DIR))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def probes_dir() -> Path:
    return PROBES_DIR


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    return GOLDEN_DIR
