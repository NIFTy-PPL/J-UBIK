"""Run every convention probe (p1..p7) + check_frames.py under pytest.

Each probe is a standalone witness that asserts its own verdict — running it
*is* the verification act (``probes/README.md``).  Here we simply execute each
one as a subprocess (repo-root cwd, the session's interpreter) and require exit
code 0.  p6/p7 need externally-minted golden fixtures; they are skipped when
those inputs have not been minted (mirroring ``check_frames.py``'s ``PENDING``
handling).

These tests intentionally do NOT re-implement the probes' logic — they pin that
the shipped probe record still passes.  The parametrized sweeps in the sibling
modules add the breadth (rectangles, anisotropy, both backends, quadrants).
"""

import subprocess
import sys
from pathlib import Path

import pytest

from conftest import GOLDEN_DIR, PROBES_DIR

# probe script -> golden inputs it requires (empty = always runnable)
PROBES = {
    "p1_metadata_vs_response.py": [],
    "p2_jwst_orientation.py": [],
    "p3_radio_orientation.py": [],
    "p4_radio_adapter.py": [],
    "p5_sky_beamer_frame.py": [],
    "p6_jwst_roundtrip.py": ["roundtrip_jwst_cal.fits"],
    "p7_radio_roundtrip.py": [
        "roundtrip_radio_obs.npz",
        "roundtrip_radio_truth.fits",
    ],
}


def _run(script: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROBES_DIR.parent),
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("name,requires", list(PROBES.items()))
def test_probe_passes(name: str, requires: list[str]) -> None:
    missing = [g for g in requires if not (GOLDEN_DIR / g).exists()]
    if missing:
        pytest.skip(f"golden input(s) not minted: {', '.join(missing)}")

    result = _run(PROBES_DIR / name)
    assert result.returncode == 0, (
        f"{name} exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def test_check_frames_consolidated() -> None:
    """check_frames.py runs every probe and asserts the documented state."""
    result = _run(PROBES_DIR / "check_frames.py")
    assert result.returncode == 0, (
        "check_frames.py reports drift from the documented convention state\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
