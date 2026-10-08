"""NIRCam: curves read from a throughput directory, and the release download."""

import tarfile

import numpy as np
import pytest
from astropy import units as u

from jubik.color import Color
from jubik.instruments.jwst.throughput.nircam import (
    download_nircam_throughputs,
    nircam_filter_weights,
)

SUFFIX = "_mean_system_throughput.txt"


def _edges(edges) -> Color:
    edges = np.asarray(edges, float)
    return Color(np.c_[edges[:-1], edges[1:]] * u.um)


def _write_curve(directory, name, wavelength_um, throughput):
    directory.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        directory / f"{name}{SUFFIX}",
        np.c_[wavelength_um, throughput],
        header="Microns Throughput",
        comments="",
    )


def _f444w_like(directory):
    """Smooth curve 3.7 to 5.1 um in the STScI file format."""
    wavelength_um = np.linspace(3.7, 5.1, 141)
    throughput = 0.5 * np.sin(np.pi * (wavelength_um - 3.7) / 1.4) ** 2
    _write_curve(directory, "F444W", wavelength_um, throughput)


def test_weights_from_the_curve_file(tmp_path):
    _f444w_like(tmp_path)
    band = nircam_filter_weights(_edges(np.linspace(3.7, 5.1, 5)), "f444w", tmp_path)
    w = band.weights
    assert band.channels == slice(0, 4) and w.shape == (4,)
    np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-12)
    assert min(w[1], w[2]) > max(w[0], w[3])
    with pytest.raises(ValueError, match="F444W"):  # red half uncovered
        nircam_filter_weights(_edges(np.linspace(3.7, 4.4, 3)), "F444W", tmp_path)


def test_missing_curve_names_the_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="F444W.*--download"):
        nircam_filter_weights(_edges([3.7, 5.1]), "F444W", tmp_path)


def test_non_nircam_filter_raises(tmp_path):
    with pytest.raises(KeyError, match="F560W"):
        nircam_filter_weights(_edges([5.0, 6.2]), "F560W", tmp_path)


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
    band = nircam_filter_weights(_edges([1.0, 1.5, 2.0]), "F150W", dest)
    np.testing.assert_allclose(
        band.weights, np.log([1.5, 2.0 / 1.5]) / np.log(2.0), rtol=1e-6
    )

    empty = tmp_path / "empty.tar.gz"
    with tarfile.open(empty, "w:gz") as tar:
        tar.add(
            source / "nircam_throughputs" / "detector_based_throughputs", arcname="x"
        )
    with pytest.raises(FileNotFoundError, match="mean_throughputs"):
        download_nircam_throughputs(tmp_path / "none", url=empty.as_uri())
