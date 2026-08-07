# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2024 Max-Planck-Society

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from numpy.testing import assert_allclose

from jubik.color import Color
from jubik.fits_saver import FitsSaver
from jubik.grid import Grid

SPATIAL_SHAPE = (4, 6)  # (x, y)
FOV = (1.0, 1.5) * u.deg
CENTER = SkyCoord(ra=10.0 * u.deg, dec=-20.0 * u.deg)


def make_grid(frequencies=None) -> Grid:
    """Grid with a well defined center, optionally with spectral bins."""
    spectral = None if frequencies is None else Color(frequencies)
    return Grid.from_shape_and_fov(
        spatial_shape=SPATIAL_SHAPE,
        fov=FOV,
        frequencies=spectral,
        sky_center=CENTER,
    )


def make_field(n_samples=1, n_pol=1, n_time=1, n_freq=1, seed=42):
    """Random field of shape (sample, polarization, time, frequency, y, x)."""
    shape = (n_samples, n_pol, n_time, n_freq, SPATIAL_SHAPE[1], SPATIAL_SHAPE[0])
    return np.random.default_rng(seed).normal(size=shape)


class TestInitialization:
    def test_accepts_six_dimensional_field(self):
        field = make_field(n_samples=3)
        saver = FitsSaver(make_grid(), field)
        assert saver.field.shape == field.shape

    @pytest.mark.parametrize("ndim", [2, 4, 5, 7])
    def test_rejects_wrong_dimensionality(self, ndim):
        field = np.zeros((1,) * ndim)
        with pytest.raises(ValueError, match="6-dimensional"):
            FitsSaver(make_grid(), field)


class TestSqueezing:
    """All non-spatial axes of length one are dropped from the written data."""

    def test_single_everything_gives_2d_image(self, tmp_path):
        field = make_field(n_samples=5)
        out = tmp_path / "mean.fits"
        FitsSaver(make_grid(), field).save_mean(str(out))

        with fits.open(out) as hdul:
            assert hdul[0].data.shape == (SPATIAL_SHAPE[1], SPATIAL_SHAPE[0])
            assert hdul[0].header["NAXIS"] == 2

    def test_samples_axis_is_kept(self, tmp_path):
        n_samples = 5
        field = make_field(n_samples=n_samples)
        out = tmp_path / "samples.fits"
        FitsSaver(make_grid(), field).save_samples(str(out))

        with fits.open(out) as hdul:
            assert hdul[0].data.shape == (
                n_samples,
                SPATIAL_SHAPE[1],
                SPATIAL_SHAPE[0],
            )
            assert hdul[0].header["CTYPE6"] == "SAMPLE"

    def test_frequency_axis_is_kept(self, tmp_path):
        grid = make_grid(frequencies=[1.0, 2.0, 3.0, 4.0] * u.Unit("GHz"))
        field = make_field(n_freq=3)
        out = tmp_path / "freq.fits"
        FitsSaver(grid, field).save_mean(str(out))

        with fits.open(out) as hdul:
            assert hdul[0].data.shape == (3, SPATIAL_SHAPE[1], SPATIAL_SHAPE[0])

    def test_infinite_single_frequency_is_squeezed(self, tmp_path):
        """A grid without spectral information has an infinite bin center."""
        grid = make_grid()
        assert np.isinf(grid.spectral.center[0])

        out = tmp_path / "nofreq.fits"
        FitsSaver(grid, make_field()).save_mean(str(out))

        with fits.open(out) as hdul:
            assert "CTYPE3" not in hdul[0].header

    def test_multiple_times_not_implemented(self, tmp_path):
        field = make_field(n_time=2)
        with pytest.raises(NotImplementedError):
            FitsSaver(make_grid(), field).save_mean(str(tmp_path / "t.fits"))

    def test_multiple_polarizations_not_implemented(self, tmp_path):
        field = make_field(n_pol=4)
        with pytest.raises(NotImplementedError):
            FitsSaver(make_grid(), field).save_mean(str(tmp_path / "p.fits"))


class TestStatistics:
    def test_mean_values(self, tmp_path):
        field = make_field(n_samples=7)
        out = tmp_path / "mean.fits"
        FitsSaver(make_grid(), field).save_mean(str(out))

        with fits.open(out) as hdul:
            assert_allclose(hdul[0].data, field.mean(axis=0).squeeze())

    def test_std_values(self, tmp_path):
        field = make_field(n_samples=7)
        out = tmp_path / "std.fits"
        FitsSaver(make_grid(), field).save_std(str(out))

        with fits.open(out) as hdul:
            assert_allclose(hdul[0].data, field.std(axis=0).squeeze())

    def test_std_bias_correction(self, tmp_path):
        n_samples = 7
        field = make_field(n_samples=n_samples)
        out = tmp_path / "std_corrected.fits"
        FitsSaver(make_grid(), field).save_std(str(out), correct_bias=True)

        expected = field.std(axis=0, ddof=1).squeeze()
        with fits.open(out) as hdul:
            assert_allclose(hdul[0].data, expected)

    def test_samples_values(self, tmp_path):
        field = make_field(n_samples=3)
        out = tmp_path / "samples.fits"
        FitsSaver(make_grid(), field).save_samples(str(out))

        with fits.open(out) as hdul:
            assert_allclose(hdul[0].data, field.squeeze())

    def test_input_field_is_not_mutated(self, tmp_path):
        field = make_field(n_samples=3)
        reference = field.copy()
        FitsSaver(make_grid(), field).save_std(
            str(tmp_path / "s.fits"), correct_bias=True
        )
        assert_allclose(field, reference)


class TestHeader:
    def test_spatial_wcs_roundtrip(self, tmp_path):
        grid = make_grid()
        out = tmp_path / "wcs.fits"
        FitsSaver(grid, make_field()).save_mean(str(out))

        with fits.open(out) as hdul:
            header = hdul[0].header

        assert "WCSAXES" not in header
        assert header["CTYPE1"] == "RA---TAN"
        assert header["CTYPE2"] == "DEC--TAN"
        assert header["RADESYS"] == "ICRS"
        assert_allclose(header["CRVAL1"], CENTER.ra.deg)
        assert_allclose(header["CRVAL2"], CENTER.dec.deg)
        assert_allclose(
            [header["CDELT1"], header["CDELT2"]],
            [
                -FOV[0].to(u.deg).value / SPATIAL_SHAPE[0],
                FOV[1].to(u.deg).value / SPATIAL_SHAPE[1],
            ],
        )

        # The written WCS maps pixels the same way as the grid it came from.
        written = WCS(header).celestial
        pixels = np.array([[0, 0], [3, 5]])
        assert_allclose(
            written.wcs_pix2world(pixels, 0),
            grid.spatial.wcs_pix2world(pixels, 0),
        )

    def test_frequency_keywords(self, tmp_path):
        # Bin bounds 1, 2, 3, 4 GHz -> centers 1.5, 2.5, 3.5 GHz.
        grid = make_grid(frequencies=[1.0, 2.0, 3.0, 4.0] * u.Unit("GHz"))
        out = tmp_path / "freq.fits"
        FitsSaver(grid, make_field(n_freq=3)).save_mean(str(out))

        with fits.open(out) as hdul:
            header = hdul[0].header

        assert header["CTYPE3"] == "FREQ"
        assert header["CUNIT3"] == "Hz"
        assert header["CRPIX3"] == 1  # FITS indexes the reference axis from 1
        assert_allclose(header["CRVAL3"], 1.5e9)
        assert_allclose(header["CDELT3"], 1.0e9)

    def test_frequency_cdelt_is_zero_for_single_channel(self, tmp_path):
        grid = make_grid(frequencies=[1.0, 2.0] * u.Unit("GHz"))
        out = tmp_path / "onefreq.fits"
        FitsSaver(grid, make_field(n_freq=1)).save_mean(str(out))

        with fits.open(out) as hdul:
            header = hdul[0].header

        # A finite single channel is described, not squeezed.
        assert header["CTYPE3"] == "FREQ"
        assert_allclose(header["CRVAL3"], 1.5e9)
        assert header["CDELT3"] == 0.0

    def test_wavelength_grid_is_converted_to_frequency(self, tmp_path):
        grid = make_grid(frequencies=[500.0, 700.0] * u.nm)
        out = tmp_path / "wl.fits"
        FitsSaver(grid, make_field(n_freq=1)).save_mean(str(out))

        expected = (600.0 * u.nm).to(u.Hz, equivalencies=u.spectral()).value
        with fits.open(out) as hdul:
            assert hdul[0].header["CUNIT3"] == "Hz"
            assert_allclose(hdul[0].header["CRVAL3"], expected)

    @pytest.mark.parametrize(
        "sky_unit, expected", [(u.Unit("Jy"), "Jy"), (u.Unit("Jy/sr"), "Jy sr-1")]
    )
    def test_bunit_is_written(self, tmp_path, sky_unit, expected):
        out = tmp_path / "bunit.fits"
        FitsSaver(make_grid(), make_field()).save_mean(str(out), sky_unit=sky_unit)

        with fits.open(out) as hdul:
            assert hdul[0].header["BUNIT"].strip() == expected

    def test_bunit_absent_without_sky_unit(self, tmp_path):
        out = tmp_path / "nobunit.fits"
        FitsSaver(make_grid(), make_field()).save_mean(str(out))

        with fits.open(out) as hdul:
            assert "BUNIT" not in hdul[0].header


class TestOverwrite:
    def test_existing_file_is_overwritten(self, tmp_path):
        out = tmp_path / "twice.fits"
        grid = make_grid()
        FitsSaver(grid, make_field(seed=1)).save_mean(str(out))

        second = make_field(seed=2)
        FitsSaver(grid, second).save_mean(str(out))

        with fits.open(out) as hdul:
            assert_allclose(hdul[0].data, second.mean(axis=0).squeeze())


class TestKnownBugs:
    """Behaviour that is wrong today; remove the xfail once fixed."""

    @pytest.mark.xfail(
        reason="_process_dynamic_axes never appends to extension_hdus, "
        "so the exact FREQUENCIES table is dropped",
        strict=True,
    )
    def test_frequencies_table_is_written(self, tmp_path):
        grid = make_grid(frequencies=[1.0, 2.0, 3.0, 4.0] * u.Unit("GHz"))
        out = tmp_path / "freqtable.fits"
        FitsSaver(grid, make_field(n_freq=3)).save_mean(str(out))

        with fits.open(out) as hdul:
            table = hdul["FREQUENCIES"]
            assert_allclose(table.data["FREQUENCY"], [1.5e9, 2.5e9, 3.5e9])

    @pytest.mark.xfail(
        reason="FITS axis numbers are assigned before squeezing, so the sample "
        "axis is described as axis 6 although the file only has 3 axes",
        strict=True,
    )
    def test_sample_axis_number_matches_naxis(self, tmp_path):
        out = tmp_path / "axisnum.fits"
        FitsSaver(make_grid(), make_field(n_samples=5)).save_samples(str(out))

        with fits.open(out) as hdul:
            header = hdul[0].header

        assert header["NAXIS"] == 3
        assert header["CTYPE3"] == "SAMPLE"
