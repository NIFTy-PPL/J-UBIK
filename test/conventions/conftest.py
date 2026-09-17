"""Shared setup for the canonical sky-frame tests.

These tests pin the claims of ``docs/source/user/canonical-sky-design.md``.
The frame every test is judged against:

    sky[i, j]:  i = dim 0  ->  +Dec (North)   [row]
                j = dim 1  ->  -RA  (West)    [column]

``test_seam_*`` files pin the named conversions in the seam table of that
page.  ``test_claims_*`` files pin, per instrument, that data and models
land in the frame above.

``JAX_PLATFORMS`` is pinned to ``cpu`` before jax is imported (transitively,
via jubik): the jaxbind ducc kernels have no GPU FFI handler.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import hashlib
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))  # glyph, radio_fixture, jwst_fixture

# The one committed data file.  CASA simulated it from the truth sky that
# ``radio_fixture.truth_hdu`` repaints; nothing in jubik can regenerate it.
# The hash pins the bytes: re-minting (``mint/mint_radio_ms.py``) must update
# it in the same commit, so a fixture can never change silently.
CASA_OBS_SHA256 = "68da2af0425c4e25fc89ac5ee99c18beadba06c1e2b60f0c23c2466fdc7f5193"


@pytest.fixture(scope="session")
def casa_observation():
    """The CASA-minted radio observation, verified against its hash.

    A missing or altered fixture is a failure, not a skip: the roundtrip
    test is the only external witness of the radio sign convention.
    """
    from radio_fixture import OBS_NPZ

    from jubik.instruments.resolve.data import Observation

    if not OBS_NPZ.exists():
        pytest.fail(f"CASA fixture missing: {OBS_NPZ}")
    digest = hashlib.sha256(OBS_NPZ.read_bytes()).hexdigest()
    if digest != CASA_OBS_SHA256:
        pytest.fail(
            f"CASA fixture {OBS_NPZ.name} has sha256 {digest}, expected "
            f"{CASA_OBS_SHA256}. If you re-minted it on purpose, update "
            "CASA_OBS_SHA256 in this conftest in the same commit."
        )
    return Observation.load(str(OBS_NPZ))


@pytest.fixture(scope="session")
def jwst_datamodel_path(tmp_path_factory):
    """Path to a synthetic JWST ImageModel with the glyph painted through its gwcs.

    Built from ``jwst_fixture`` at session start.  Skips only when the
    optional ``jwst`` dependency is not installed.
    """
    pytest.importorskip("jwst", reason="jwst extra not installed")
    import jwst_fixture

    path = tmp_path_factory.mktemp("jwst") / "roundtrip_cal.fits"
    jwst_fixture.build_datamodel().save(str(path))
    return path
