"""Tests for `jubik.instruments.hst.tinytim`.

`tiny1` and `tiny2` are fake executables in a temporary TINYTIM directory: they
log their command line and answers and write a Gaussian "PSF", so Tiny Tim
need not be installed. The last test runs the real Tiny Tim if it is.
"""

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
from astropy.io import fits

from jubik.instruments.hst.tinytim import load_hst_psf_kernel, tinytim_kernel

FAKE_TINY1 = """#!{python}
import json, os, sys

log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calls.jsonl")
fail = os.environ.get("FAKE_TINYTIM_FAIL", "")
if sys.argv[1] == "-v":
    print(os.environ.get("FAKE_TINYTIM_VERSION", "7.5"))
    sys.exit(0)
if fail == "tiny1-loop":
    while True:
        print("Choice : ")
answers = sys.stdin.read().split("\\n")[:-1]
spectrum = open("spectrum.dat").read() if os.path.exists("spectrum.dat") else None
with open(log, "a") as f:
    f.write(json.dumps(["tiny1", sys.argv[1:], answers, spectrum]) + "\\n")
if fail == "tiny1-exit":
    print("** ERROR : That filter is not in the database **")
    sys.exit(2)
# answers end with: diameter, N | Y, factor, despace, rootname
if answers[-3] == "N":
    subsample, diameter = 1, float(answers[-4])
else:
    subsample, diameter = int(answers[-3]), float(answers[-5])
n = int(diameter / 0.05)
n = (n - n % 2) * subsample
with open(sys.argv[1], "w") as f:
    f.write(f"{{n}}  # Integrated PSF dimension (pixels)\\n")
    f.write(f"{{0.05 / subsample:f}}  # Integrated PSF scaling (arcsec)\\n")
"""

FAKE_TINY2 = """#!{python}
import json, os, sys

import numpy as np
from astropy.io import fits

log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calls.jsonl")
fail = os.environ.get("FAKE_TINYTIM_FAIL", "")
with open(log, "a") as f:
    f.write(json.dumps(["tiny2", sys.argv[1:]]) + "\\n")
print("Integrated PSF dimensions are ...")
if fail == "tiny2-exit":
    print("FAKE TINY2 FAILURE")
    sys.exit(3)
if fail == "tiny2-no-output":
    sys.exit(0)
n = int(open(sys.argv[1]).readline().split()[0])
# Centred on pixel n/2 of the even grid, as Tiny Tim does; sums to 0.97.
y, x = np.mgrid[:n, :n] - n // 2
psf = np.exp(-(x**2 + y**2) / (2 * 1.5**2))
if fail == "tiny2-zeros":
    psf[:] = 0
fits.writeto("psf00.fits", (0.97 * psf / psf.sum()).astype(np.float32))
"""


@pytest.fixture
def fake_tinytim(tmp_path, monkeypatch):
    """A fake Tiny Tim install. Returns a function listing the logged calls."""
    home = tmp_path / "tinytim"
    home.mkdir()
    for name, script in (("tiny1", FAKE_TINY1), ("tiny2", FAKE_TINY2)):
        path = home / name
        path.write_text(script.format(python=sys.executable))
        path.chmod(0o755)
    (home / "wfpc2.fil").write_text("F160BW 29\nF555W 23\nF656N 1\nF814W 18\n")
    (home / "nicmos1.fil").write_text("F110M 12\nF110W 31\nF160W 15\n")
    (home / "nicmos2.fil").write_text("F110W 31\nF160W 16\n")
    monkeypatch.setenv("TINYTIM", str(home))
    monkeypatch.delenv("FAKE_TINYTIM_FAIL", raising=False)
    monkeypatch.delenv("FAKE_TINYTIM_VERSION", raising=False)

    def calls():
        log = home / "calls.jsonl"
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines()]

    return calls


@pytest.fixture
def library(tmp_path):
    return str(tmp_path / "psf_library")


def test_wfpc2_answers_follow_tiny1_prompts(fake_tinytim, library):
    tinytim_kernel("WFPC2", "F814W", (400.4, 399.6), library, chip=3)
    tinytim_kernel("WFPC2", "F555W", (10, 20), library, chip=1, subsample=5)
    tinytim_kernel("WFPC2", "f656n", (10, 20), library, chip=4)

    # tiny1.c (7.5): camera, chip (WF only), position, filter, spectrum form
    # and index (only for filters of several wavelengths), diameter,
    # subsampled?, factor, despace, rootname.
    assert fake_tinytim() == [
        [
            "tiny1",
            ["tiny.param"],
            ["5", "3", "400 400", "F814W", "3", "0", "3.0", "N", "0.0", "psf"],
            None,
        ],
        ["tiny2", ["tiny.param"]],
        [
            "tiny1",
            ["tiny.param"],
            ["6", "10 20", "F555W", "3", "0", "3.0", "Y", "5", "0.0", "psf"],
            None,
        ],
        ["tiny2", ["tiny.param"]],
        [
            "tiny1",
            ["tiny.param"],
            ["5", "4", "10 20", "F656N", "3.0", "N", "0.0", "psf"],
            None,
        ],
        ["tiny2", ["tiny.param"]],
    ]


@pytest.mark.parametrize(
    "camera, date, expected",
    [
        # Before the cryocooler, tiny1 asks the date for the aberrations.
        ("NIC1", "1997-09-12", ["9", "128 64", "12 9 1997"]),
        ("NIC2", "1998-01-02T03:04:05", ["10", "128 64", "2 1 1998"]),
        # With it, whether the date is after 16 May (NIC1) or 29 Sep 2002 (NIC2).
        ("NIC1", "2002-06-01", ["19", "128 64", "Y"]),
        ("NIC2", "2002-06-01", ["20", "128 64", "N"]),
        ("NIC2", "2005-06-01", ["20", "128 64", "Y"]),
    ],
)
def test_nicmos_answers_follow_tiny1_prompts(
    fake_tinytim, library, tmp_path, camera, date, expected
):
    spectrum = tmp_path / "my spectrum.txt"
    spectrum.write_text("FNU\n10000 1.0\n20000 2.0\n")

    tinytim_kernel(
        camera,
        "F160W",
        (128, 64),
        library,
        date=date,
        spectrum=str(spectrum),
        psf_arcsec=2,
        despace=1.5,
        jitter=7,
    )

    [(stage, args, answers, copied_spectrum), _] = fake_tinytim()
    assert args == ["tiny.param", "jitter=7.0"]
    assert answers == expected + [
        "F160W",
        "5",
        "spectrum.dat",
        "2.0",
        "N",
        "1.5",
        "psf",
    ]
    assert copied_spectrum == spectrum.read_text()


def test_cache_hit_runs_tiny_tim_once(fake_tinytim, library, monkeypatch):
    first = tinytim_kernel("NIC1", "F160W", (128, 128), library, date="1997-09-12")
    # A hit needs no Tiny Tim, and Tiny Tim's filter names are case-insensitive.
    monkeypatch.delenv("TINYTIM")
    second = tinytim_kernel("NIC1", "f160w", (128, 128), library, date="1997-09-12")

    assert [c[0] for c in fake_tinytim()] == ["tiny1", "tiny2"]
    np.testing.assert_array_equal(first.kernel, second.kernel)
    assert (first.pixel_scale, first.subsample, first.charge_diffusion) == (
        second.pixel_scale,
        second.subsample,
        second.charge_diffusion,
    )
    # The run directory is removed after a successful run.
    assert [p for p in os.listdir(library) if p.startswith("tinytim_")] == []


NIC1 = dict(camera="NIC1", filter="F160W", position=(128, 128), date="1997-09-12")
WFPC2 = dict(camera="WFPC2", filter="F555W", position=(400, 400), chip=2)


@pytest.mark.parametrize(
    "base, change",
    [
        (NIC1, dict(camera="NIC2")),
        (WFPC2, dict(chip=3)),
        (NIC1, dict(filter="F110W")),
        (NIC1, dict(position=(128, 129))),
        (NIC1, dict(date="1998-09-12")),
        (NIC1, dict(spectrum="FNU\n10000 1.0\n")),
        (dict(NIC1, spectrum="FNU\n10000 1.0\n"), dict(spectrum="FNU\n10000 2.0\n")),
        (NIC1, dict(subsample=2)),
        (NIC1, dict(psf_arcsec=2.0)),
        (NIC1, dict(despace=1.0)),
        (NIC1, dict(jitter=5.0)),
    ],
)
def test_cache_misses_when_one_input_changes(
    fake_tinytim, library, tmp_path, base, change
):
    def call(kwargs):
        kwargs = dict(kwargs)
        if "spectrum" in kwargs:
            path = tmp_path / "spectrum.txt"
            path.write_text(kwargs["spectrum"])
            kwargs["spectrum"] = str(path)
        return tinytim_kernel(psf_library_path=library, **kwargs)

    call(base)
    call(dict(base, **change))

    assert [c[0] for c in fake_tinytim()].count("tiny2") == 2
    assert len(os.listdir(library)) == 2


@pytest.mark.parametrize(
    "kwargs, scale, charge_diffusion",
    [
        (dict(WFPC2), 0.05, True),
        (dict(WFPC2, subsample=4), 0.0125, False),
        (dict(NIC1, subsample=3), 0.05 / 3, False),
        (dict(NIC1), 0.05, False),
    ],
)
def test_kernel_is_normalized_and_centred(
    fake_tinytim, library, kwargs, scale, charge_diffusion
):
    psf = tinytim_kernel(psf_library_path=library, **kwargs)

    kernel = psf.kernel
    assert all(size % 2 for size in kernel.shape)
    np.testing.assert_allclose(kernel.sum(), 1.0, rtol=1e-12)
    middle = (kernel.shape[0] - 1) / 2
    y, x = np.indices(kernel.shape)
    np.testing.assert_allclose(
        [(y * kernel).sum(), (x * kernel).sum()], [middle, middle], atol=1e-6
    )
    assert np.unravel_index(kernel.argmax(), kernel.shape) == (middle, middle)
    # tiny1 writes the scale with 6 decimals.
    np.testing.assert_allclose(psf.pixel_scale, scale, rtol=1e-4)
    assert psf.subsample == kwargs.get("subsample", 1)
    assert psf.charge_diffusion is charge_diffusion


@pytest.mark.parametrize(
    "fail, message",
    [
        ("tiny1-exit", "tiny1 exited with 2 and wrote no tiny.param"),
        ("tiny1-loop", "tiny1 exited with -9"),
        ("tiny2-exit", "tiny2 exited with 3 and wrote no psf00.fits"),
        ("tiny2-no-output", "tiny2 exited with 0 and wrote no psf00.fits"),
    ],
)
def test_failure_keeps_run_directory_and_shows_output(
    fake_tinytim, library, monkeypatch, fail, message
):
    monkeypatch.setenv("FAKE_TINYTIM_FAIL", fail)

    with pytest.raises(RuntimeError, match=message) as error:
        tinytim_kernel(psf_library_path=library, **WFPC2)

    stage = fail.split("-")[0]
    [run_dir] = [p for p in os.listdir(library) if p.startswith("tinytim_")]
    assert os.path.join(library, run_dir) in str(error.value)
    assert {"tiny1.in", f"{stage}.log"} <= set(
        os.listdir(os.path.join(library, run_dir))
    )
    expected_tail = {
        "tiny1-exit": "That filter is not in the database",
        "tiny1-loop": "Choice :",
        "tiny2-exit": "FAKE TINY2 FAILURE",
        "tiny2-no-output": "Integrated PSF dimensions are",
    }[fail]
    assert expected_tail in str(error.value)
    assert not [p for p in os.listdir(library) if p.endswith(".fits")]


def test_not_installed_says_how_to_install(
    fake_tinytim, library, tmp_path, monkeypatch
):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("TINYTIM")
    with pytest.raises(FileNotFoundError, match="TINYTIM=''.*make linux"):
        tinytim_kernel(psf_library_path=library, **WFPC2)

    monkeypatch.setenv("TINYTIM", str(tmp_path))
    with pytest.raises(FileNotFoundError, match=f"TINYTIM='{tmp_path}'"):
        tinytim_kernel(psf_library_path=library, **WFPC2)

    # Tiny Tim would look for its tables relative to the run directory.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TINYTIM", "tinytim")
    with pytest.raises(FileNotFoundError, match="TINYTIM='tinytim'.*absolute"):
        tinytim_kernel(psf_library_path=library, **WFPC2)
    assert fake_tinytim() == []


def test_invalid_output_is_not_cached(fake_tinytim, library, monkeypatch):
    monkeypatch.setenv("FAKE_TINYTIM_FAIL", "tiny2-zeros")

    with pytest.raises(ValueError, match="kernel sum is nan"):
        tinytim_kernel(psf_library_path=library, **WFPC2)

    [run_dir] = os.listdir(library)
    assert run_dir.startswith("tinytim_")
    assert "tiny2.log" in os.listdir(os.path.join(library, run_dir))


def test_other_tinytim_version_is_refused(fake_tinytim, library, monkeypatch):
    monkeypatch.setenv("FAKE_TINYTIM_VERSION", "6.3")
    with pytest.raises(RuntimeError, match=r"(?s)scripted for Tiny Tim 7\.5.*6\.3"):
        tinytim_kernel(psf_library_path=library, **WFPC2)
    assert fake_tinytim() == []


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        (dict(WFPC2, chip=None), ValueError, "chip"),
        (dict(WFPC2, chip=2.0), TypeError, "float"),
        (dict(NIC1, date=None), ValueError, "date"),
        (dict(NIC1, camera="ACS"), ValueError, "camera"),
        (dict(NIC1, position=(256, 0)), ValueError, "off the 256x256"),
        (dict(NIC1, subsample=11), ValueError, "subsample"),
        (dict(NIC1, subsample=2.0), TypeError, "float"),
        (dict(WFPC2, filter="F656N", spectrum="s.txt"), ValueError, "monochromatic"),
        # tiny1 would match F110M only on NIC1, and F160 to F160BW.
        (dict(NIC1, camera="NIC2", filter="F110M"), ValueError, "not a NIC2 filter"),
        (dict(WFPC2, filter="F160"), ValueError, "not a WFPC2 filter"),
    ],
)
def test_invalid_inputs_are_refused(fake_tinytim, library, kwargs, error, message):
    with pytest.raises(error, match=message):
        tinytim_kernel(psf_library_path=library, **kwargs)
    assert fake_tinytim() == []


def test_load_prepared_kernel(tmp_path):
    hdu = fits.PrimaryHDU(np.arange(15.0).reshape(3, 5))
    hdu.header.update(PIXSCALE=0.0432, SUBSAMPL=1, CHGDIFF=False)
    hdu.writeto(tmp_path / "prepared.fits")

    psf = load_hst_psf_kernel(str(tmp_path / "prepared.fits"))

    np.testing.assert_allclose(psf.kernel, np.arange(15.0).reshape(3, 5) / 105)
    assert (psf.pixel_scale, psf.subsample, psf.charge_diffusion) == (0.0432, 1, False)


@pytest.mark.parametrize(
    "data, header, error",
    [
        (np.ones((3, 3)), dict(PIXSCALE=0.05, SUBSAMPL=1), KeyError),
        (np.ones((3, 3)), dict(PIXSCALE=0.05, SUBSAMPL=1, CHGDIFF="F"), ValueError),
        (np.ones((4, 4)), dict(PIXSCALE=0.05, SUBSAMPL=1, CHGDIFF=False), ValueError),
        (np.zeros((3, 3)), dict(PIXSCALE=0.05, SUBSAMPL=1, CHGDIFF=False), ValueError),
    ],
)
def test_load_refuses_kernels_without_metadata_or_centre(tmp_path, data, header, error):
    hdu = fits.PrimaryHDU(data)
    hdu.header.update(header)
    hdu.writeto(tmp_path / "prepared.fits")
    with pytest.raises(error):
        load_hst_psf_kernel(str(tmp_path / "prepared.fits"))


TINY1 = shutil.which(
    "tiny1",
    path=os.pathsep.join((os.environ.get("TINYTIM", ""), os.environ.get("PATH", ""))),
)


@pytest.mark.skipif(
    not os.environ.get("TINYTIM") or TINY1 is None, reason="Tiny Tim is not installed"
)
def test_matches_direct_tinytim_run(tmp_path):
    psf = tinytim_kernel(
        "WFPC2", "F555W", (400, 400), str(tmp_path / "library"), chip=1, psf_arcsec=1.0
    )

    # The same run by hand, answering tiny1's prompts as in the manual.
    tiny2 = os.path.join(os.path.dirname(TINY1), "tiny2")
    answers = "6\n400 400\nF555W\n3\n0\n1.0\nN\n0.0\ndirect\n"
    subprocess.run(
        [TINY1, "direct.param"], input=answers, text=True, cwd=tmp_path, check=True
    )
    subprocess.run([tiny2, "direct.param"], cwd=tmp_path, check=True)
    direct = fits.getdata(tmp_path / "direct00.fits").astype(float)

    np.testing.assert_allclose(psf.kernel[:-1, :-1], direct / direct.sum(), rtol=1e-6)
    middle = (psf.kernel.shape[0] - 1) // 2
    assert np.unravel_index(psf.kernel.argmax(), psf.kernel.shape) == (middle, middle)
    assert psf.charge_diffusion
    np.testing.assert_allclose(psf.pixel_scale, 0.0455, rtol=0.01)
