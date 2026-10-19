"""Tests for `jubik.instruments.jwst.psf.jwst_kernel`.

A recording `stpsf` stub is put into `sys.modules`, so neither the real stpsf
nor its calibration data are touched and nothing is downloaded.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from jubik.instruments.jwst.parse.jwst_psf import JwstPsfKernelConfig
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
        "NIRCAM",
        "f444w",
        (10.0, 20.0),
        calibration_dir,
        2,
        0.4,
        "last",
        detector="NRCBLONG",
    )

    inits = [c for c in stpsf_stub if c[0] == "init"]
    assert inits == [("init", "NIRCam", calibration_dir)]
    assert ("set", "filter", "F444W") in stpsf_stub
    np.testing.assert_array_equal(psf, np.ones((3, 3)))


def test_build_webb_psf_unknown_camera_mentions_stpsf(stpsf_stub):
    with pytest.raises(KeyError, match="STPSF"):
        jwst_kernel.build_webb_psf("NIRSPEC", "f444w", (0, 0), "", 1, detector="NRS1")
    assert not [c for c in stpsf_stub if c[0] == "init"]


def make_jwst_data(camera, detector, local_pixel, xstart=1, ystart=1):
    return SimpleNamespace(
        camera=camera,
        filter="F444W" if camera == "NIRCAM" else "F770W",
        wcs=SimpleNamespace(world_to_pixel=lambda _: local_pixel),
        dm=SimpleNamespace(
            meta=SimpleNamespace(
                instrument=SimpleNamespace(detector=detector),
                subarray=SimpleNamespace(xstart=xstart, ystart=ystart),
            )
        ),
    )


def make_psf_config(tmp_path, normalize="last"):
    return JwstPsfKernelConfig(
        webbpsf_path=str(tmp_path / "stpsf-data"),
        psf_library_path=str(tmp_path / "psf_library"),
        psf_arcsec=0.4,
        normalize=normalize,
    )


def detector_settings(calls):
    return [c[1:] for c in calls if c[0] == "set" and c[1].startswith("detector")]


@pytest.mark.parametrize("detector", ["NRCALONG", "NRCBLONG"])
def test_load_psf_kernel_selects_observation_detector(stpsf_stub, tmp_path, detector):
    data = make_jwst_data("NIRCAM", detector, (100.0, 200.0))

    jwst_kernel.load_psf_kernel(data, 2, None, make_psf_config(tmp_path))

    assert detector_settings(stpsf_stub) == [
        ("detector", detector),
        ("detector_position", (100.0, 200.0)),
    ]


def test_load_psf_kernel_maps_miri_detector_name(stpsf_stub, tmp_path):
    data = make_jwst_data("MIRI", "MIRIMAGE", (100.0, 200.0))

    jwst_kernel.load_psf_kernel(data, 2, None, make_psf_config(tmp_path))

    assert ("init", "MIRI", str(tmp_path / "stpsf-data")) in stpsf_stub
    assert detector_settings(stpsf_stub)[0] == ("detector", "MIRIM")


def test_load_psf_kernel_uses_full_detector_position(stpsf_stub, tmp_path):
    # SUB640 style subarray with a 1-based origin of (1025, 513).
    data = make_jwst_data("NIRCAM", "NRCB1", (10.5, 20.5), xstart=1025, ystart=513)

    jwst_kernel.load_psf_kernel(data, 2, None, make_psf_config(tmp_path))

    assert detector_settings(stpsf_stub)[-1] == ("detector_position", (1034.5, 532.5))
