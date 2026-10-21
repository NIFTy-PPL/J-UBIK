"""NIRCam: curves read from a throughput directory, and the release download."""

import tarfile

import numpy as np
import pytest
from astropy import units as u

from jubik.instruments.jwst.throughput.nircam import (
    download_nircam_throughputs,
    nircam_throughput,
)

SUFFIX = "_mean_system_throughput.txt"


def _write_curve(directory, name, wavelength_um, throughput):
    directory.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        directory / f"{name}{SUFFIX}",
        np.c_[wavelength_um, throughput],
        header="Microns Throughput",
        comments="",
    )


def test_curve_from_the_file(tmp_path):
    wavelength_um = np.linspace(3.7, 5.1, 141)
    throughput = 0.5 * np.sin(np.pi * (wavelength_um - 3.7) / 1.4) ** 2
    _write_curve(tmp_path, "F444W", wavelength_um, throughput)
    curve = nircam_throughput("f444w", tmp_path)
    assert curve.wavelength.unit == u.um
    np.testing.assert_allclose(curve.wavelength.value, wavelength_um)
    np.testing.assert_allclose(curve.throughput, throughput)


def test_release_noise_below_zero_is_clipped(tmp_path):
    # the two F150W2 samples from nircam_throughputs_4Nov2022_v5
    wavelength_um = [0.790, 0.797, 0.824, 0.850, 1.0, 2.0, 2.4]
    throughput = [
        0.0,
        -0.00008562228180158444,
        -0.00037624569375188134,
        0.0,
        0.5,
        0.5,
        0.0,
    ]
    _write_curve(tmp_path, "F150W2", wavelength_um, throughput)
    curve = nircam_throughput("F150W2", tmp_path)
    np.testing.assert_array_equal(curve.throughput[:4], 0.0)
    np.testing.assert_allclose(curve.throughput[4:], [0.5, 0.5, 0.0])


def test_negative_beyond_noise_raises(tmp_path):
    _write_curve(tmp_path, "F150W2", [1.0, 1.5, 2.0], [0.0, -0.01, 1.0])
    with pytest.raises(ValueError, match="F150W2.*-0.01.*not interpolation noise"):
        nircam_throughput("F150W2", tmp_path)


def test_missing_curve_names_the_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="F444W.*--download"):
        nircam_throughput("F444W", tmp_path)


def test_non_nircam_filter_raises(tmp_path):
    with pytest.raises(KeyError, match="F560W"):
        nircam_throughput("F560W", tmp_path)


def test_download_unpacks_the_mean_curves(tmp_path):
    # a release archive with the STScI layout, served from a file:// url
    source = tmp_path / "src"
    _write_curve(
        source / "nircam_throughputs" / "mean_throughputs",
        "F150W",
        [1.0, 2.0],
        [1.0, 1.0],
    )
    _write_curve(
        source / "nircam_throughputs" / "mean_throughputs",
        "F200W",
        [1.5, 2.5],
        [1.0, 1.0],
    )
    _write_curve(
        source / "nircam_throughputs" / "detector_based_throughputs",
        "NRCA1_F150W",
        [1.0, 2.0],
        [1.0, 1.0],
    )
    (
        source / "nircam_throughputs" / "mean_throughputs" / "F150W_summary.pdf"
    ).write_bytes(b"pdf")
    archive = tmp_path / "release.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source / "nircam_throughputs", arcname="nircam_throughputs")

    dest = tmp_path / "throughputs"
    assert download_nircam_throughputs(dest, url=archive.as_uri()) == 2
    assert sorted(p.name for p in dest.iterdir()) == [
        f"F150W{SUFFIX}",
        f"F200W{SUFFIX}",
    ]
    curve = nircam_throughput("F200W", dest)
    np.testing.assert_allclose(curve.wavelength.to_value(u.um), [1.5, 2.5])
    np.testing.assert_allclose(curve.throughput, [1.0, 1.0])

    empty = tmp_path / "empty.tar.gz"
    with tarfile.open(empty, "w:gz") as tar:
        tar.add(
            source / "nircam_throughputs" / "detector_based_throughputs", arcname="x"
        )
    with pytest.raises(FileNotFoundError, match="mean_throughputs"):
        download_nircam_throughputs(tmp_path / "none", url=empty.as_uri())
