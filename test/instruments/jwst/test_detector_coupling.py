"""NIRCam detector coupling (STPSF's IPC and PPC) in the kernel and the star fit."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.signal import convolve

from jubik.instruments.jwst.parse.jwst_psf import JwstPsfKernelConfig
from jubik.instruments.jwst.psf import jwst_kernel
from jubik.instruments.jwst.psf.jwst_kernel import (
    oversampled_coupling,
    with_detector_coupling,
)
from jubik.instruments.jwst.psf.psf_correction import (
    PsfCorrection,
    StarStamp,
    _StarModel,
    fit_psf_correction,
    with_pixel_box,
)

from test_jwst_likelihoods import FakeJwstData, patched_seams  # noqa: F401
from test_psf_correction import SIZE, SUB, kernel, mock_stars, texture

RNG = np.random.default_rng(486)


def near_delta(rng):
    """A 3 x 3 detector-pixel kernel summing to one, asymmetric as PPC is."""
    k = rng.uniform(0, 0.02, (3, 3))
    k[1, 1] = 0
    k[1, 1] = 1 - k.sum()
    return k


def binned(image, s):
    n = image.shape[0] // s
    return image.reshape(n, s, n, s).sum((1, 3))


@pytest.mark.parametrize("s", [2, 3])
def test_coupling_then_binning_equals_binning_then_coupling(s):
    """The comb at detector pitch acts on whole pixels: binning commutes."""
    ipc, ppc = near_delta(RNG), near_delta(RNG)
    image = RNG.uniform(0, 1, (12 * s, 12 * s))
    coupled = binned(
        with_detector_coupling(image, oversampled_coupling(ipc, ppc, s)), s
    )
    expected = convolve(binned(image, s), convolve(ipc, ppc), mode="same")
    np.testing.assert_allclose(coupled, expected, rtol=0, atol=1e-13)


def test_coupling_keeps_the_kernel_sum_and_moves_ppc_along_x():
    ppc = np.zeros((3, 3))
    ppc[1, 1], ppc[1, 2] = 0.9, 0.1  # STPSF's NIRCam PPC: charge to the next pixel in x
    coupling = oversampled_coupling(np.pad([[1.0]], 1), ppc, SUB)
    psf = np.pad(kernel(), 10)  # room for the coupling
    coupled = with_detector_coupling(psf, coupling)
    np.testing.assert_allclose(coupled.sum(), psf.sum(), rtol=1e-12)
    np.testing.assert_allclose(
        coupled, 0.9 * psf + 0.1 * np.roll(psf, SUB, 1), atol=1e-15
    )


def test_stpsf_reads_the_detectors_coupling(monkeypatch):
    """The long-wavelength names reach STPSF by module number."""
    import stpsf.detectors

    class Kernel:
        def __init__(self, a):
            self.array = a

    ipc, ppc, asked = near_delta(RNG), near_delta(RNG), []

    def fake(inst, header):
        asked.append((inst, header["DET_NAME"]))
        return (Kernel(ipc), Kernel(ppc)), {}

    monkeypatch.setattr(stpsf.detectors, "get_detector_ipc_model", fake)
    out = jwst_kernel.detector_coupling_kernel("nrcblong", SUB, "stpsf-data")
    np.testing.assert_array_equal(out, oversampled_coupling(ipc, ppc, SUB))
    jwst_kernel.detector_coupling_kernel("NRCB1", SUB, "stpsf-data")
    assert asked == [("NIRCAM", "NRCB5"), ("NIRCAM", "NRCB1")]


@pytest.fixture
def fake_coupling(monkeypatch):
    coupling = oversampled_coupling(near_delta(RNG), near_delta(RNG), 2)
    asked = []

    def fake(detector, subsample, webbpsf_path):
        asked.append((detector, subsample))
        return coupling

    monkeypatch.setattr(jwst_kernel, "detector_coupling_kernel", fake)
    return coupling, asked


def test_load_psf_kernel_couples_after_the_correction_and_box(
    patched_seams, fake_coupling, tmp_path
):
    coupling, asked = fake_coupling
    raw = {"webbpsf_path": "", "psf_library": str(tmp_path)}
    raw["psf_arcsec_extension"] = 0.4
    data, centre = FakeJwstData("x.fits"), FakeJwstData("x.fits").wcs.center
    config = JwstPsfKernelConfig.from_yaml_dict(raw)
    assert config.detector_coupling is False
    plain = jwst_kernel.load_psf_kernel(data, 2, centre, config)
    assert not asked
    coupled = JwstPsfKernelConfig.from_yaml_dict(raw | {"detector_coupling": True})
    np.testing.assert_array_equal(
        jwst_kernel.load_psf_kernel(data, 2, centre, coupled),
        with_detector_coupling(plain, coupling),
    )
    assert asked == [("NRCALONG", 2)]

    t = np.zeros((3, 3))
    t[1, 1] = 1.0
    for on in (True, False):
        c = PsfCorrection("f444w", 2, t, 0.3, pixel_box=True, detector_coupling=on)
        c.save(tmp_path / f"{on}.npz")
    files = {on: str(tmp_path / f"{on}.npz") for on in (True, False)}
    both = raw | {"pixel_box": True, "detector_coupling": True}
    config = JwstPsfKernelConfig.from_yaml_dict(
        both | {"correction": {"f444w": files[True]}}
    )
    expected = with_detector_coupling(
        with_pixel_box(PsfCorrection.load(files[True]).apply(plain)), coupling
    )
    np.testing.assert_array_equal(
        jwst_kernel.load_psf_kernel(data, 2, centre, config), expected
    )
    for on in (True, False):
        mismatch = raw | {
            "pixel_box": True,
            "detector_coupling": on,
            "correction": {"f444w": files[not on]},
        }
        with pytest.raises(ValueError, match="learned with detector_coupling"):
            jwst_kernel.load_psf_kernel(
                data, 2, centre, JwstPsfKernelConfig.from_yaml_dict(mismatch)
            )
    data.camera = "MIRI"
    with pytest.raises(ValueError, match="for NIRCam, not miri"):
        jwst_kernel.load_psf_kernel(data, 2, centre, coupled)


def test_save_keeps_the_coupling_flag_only_when_set(tmp_path):
    for on in (True, False):
        PsfCorrection("f", SUB, np.zeros((1, 1)), 0.4, detector_coupling=on).save(
            tmp_path / "c.npz"
        )
        with np.load(tmp_path / "c.npz") as f:
            assert ("detector_coupling" in f.files) == on
        assert PsfCorrection.load(tmp_path / "c.npz").detector_coupling is on


@pytest.mark.parametrize("k", [40, 41])
def test_star_model_couples_as_the_loader_does(k):
    """What the star fit learns with a stamp's coupling is what
    `load_psf_kernel` gives the lens fit: the coupling after the texture, the
    width and the box (an asymmetric coupling: its direction counts)."""
    base, coupling = kernel(k), oversampled_coupling(
        near_delta(RNG), near_delta(RNG), SUB
    )
    c = PsfCorrection("f", SUB, texture(9), 0.8)
    applied = with_detector_coupling(with_pixel_box(c.apply(base)), coupling)
    ones = np.ones((SIZE, SIZE))
    learning = _StarModel(
        [StarStamp(ones, ones, ones.astype(bool), base, coupling=coupling)],
        SUB,
        pixel_box=True,
    )
    frozen = _StarModel([StarStamp(ones, ones, ones.astype(bool), applied)], SUB)
    learning.flux0 = frozen.flux0 = np.array([1e4])
    x = {key: np.full(v.shape, 0.3) for key, v in learning.domain.items()}
    star = np.asarray(frozen(x)) - frozen(x | {"star_flux": np.full(1, -np.inf)})
    np.testing.assert_allclose(
        learning(x, c.log_texture, c.width), frozen(x), rtol=0, atol=1e-6 * star.max()
    )


def test_every_stamp_or_none_and_the_fit_records_it():
    coupling = oversampled_coupling(near_delta(RNG), near_delta(RNG), SUB)
    stamps = mock_stars(kernel(), PsfCorrection("f", SUB, np.zeros((1, 1)), 0.5), 4)
    coupled = [replace(s, coupling=coupling) for s in stamps]
    with pytest.raises(ValueError, match="every StarStamp has a coupling or none"):
        _StarModel(coupled[:2] + stamps[2:], SUB)
    with pytest.raises(ValueError, match="square, odd"):
        _StarModel([replace(s, coupling=np.ones((4, 4))) for s in stamps], SUB)
    for these, on in ((coupled, True), (stamps, False)):
        learned, _ = fit_psf_correction(
            these, SUB, "f", width=(0.5, 0.5), n_iterations=1
        )
        assert learned.detector_coupling is on
