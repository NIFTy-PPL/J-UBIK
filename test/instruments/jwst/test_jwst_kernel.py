"""Tests for `jubik.instruments.jwst.psf.jwst_kernel`.

A recording `stpsf` stub is put into `sys.modules`, so neither the real stpsf
nor its calibration data are touched and nothing is downloaded.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from jubik.instruments.jwst.psf import jwst_kernel


def make_stpsf_stub():
    calls = []

    class RecordingInstrument:
        def __init__(self):
            calls.append(("init", type(self).__name__, os.environ.get("STPSF_PATH")))

        def __setattr__(self, name, value):
            calls.append(("set", name, value))
            super().__setattr__(name, value)

        def calc_psf(self, **kwargs):
            calls.append(("calc_psf", kwargs))
            return [None, None, SimpleNamespace(data=np.ones((3, 3)))]

    stub = SimpleNamespace(
        NIRCam=type("NIRCam", (RecordingInstrument,), {}),
        MIRI=type("MIRI", (RecordingInstrument,), {}),
    )
    return stub, calls


@pytest.fixture
def stpsf_stub(monkeypatch):
    monkeypatch.delenv("STPSF_PATH", raising=False)
    monkeypatch.delenv("WEBBPSF_PATH", raising=False)
    stub, calls = make_stpsf_stub()
    monkeypatch.setitem(sys.modules, "stpsf", stub)
    return calls


def test_build_webb_psf_passes_path_and_inits_only_selected_instrument(
    stpsf_stub, tmp_path
):
    calibration_dir = str(tmp_path / "stpsf-data")

    psf = jwst_kernel.build_webb_psf(
        "NIRCAM", "f444w", (10.0, 20.0), calibration_dir, 2, 0.4, "last"
    )

    inits = [c for c in stpsf_stub if c[0] == "init"]
    assert inits == [("init", "NIRCam", calibration_dir)]
    assert ("set", "filter", "F444W") in stpsf_stub
    np.testing.assert_array_equal(psf, np.ones((3, 3)))


def test_build_webb_psf_unknown_camera_mentions_stpsf(stpsf_stub):
    with pytest.raises(KeyError, match="STPSF"):
        jwst_kernel.build_webb_psf("NIRSPEC", "f444w", (0, 0), "", 1)
    assert not [c for c in stpsf_stub if c[0] == "init"]
