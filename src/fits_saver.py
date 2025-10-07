from typing import Optional
import numpy as np
from astropy.io import fits
from astropy import units as u
from jubik0.grid import Grid
from numpy.typing import NDArray

__all__ = ["FitsSaver"]

# --------------------------------------------------------------------------------------
# Private Subroutines ------------------------------------------------------------------
# --------------------------------------------------------------------------------------


def _process_frequency(
    header: fits.Header, grid: Grid, field: NDArray, np_axis: int, fits_axis: int
) -> tuple[Optional[fits.BinTableHDU], NDArray]:
    """Processes the frequency axis. This axis is never squeezed."""
    freqs = u.Quantity(grid.spectral.center).to(u.Hz, equivalencies=u.spectral())

    if field.shape[np_axis] == 1 and np.isinf(freqs[0]):
        return None, np.squeeze(field, axis=np_axis)

    # This is only approximate for backward compatibility
    header[f"CTYPE{fits_axis}"] = "FREQ"
    header[f"CUNIT{fits_axis}"] = freqs.unit.to_string("fits")
    header[f"CRPIX{fits_axis}"] = 1  # Fits reference 0-th axis is indexed by 1
    header[f"CRVAL{fits_axis}"] = freqs[0].value
    header[f"CDELT{fits_axis}"] = (
        (freqs[1].value - freqs[0].value) if len(freqs) > 1 else 0.0
    )

    # This is the most exact description only available in newer programs.
    freq_col = fits.Column(
        name="FREQUENCY",
        format="E",
        unit=freqs.unit.to_string("fits"),
        array=freqs.value,
    )
    return fits.BinTableHDU.from_columns([freq_col], name="FREQUENCIES"), field


def _process_time(
    header: fits.Header, grid: Grid, field: NDArray, np_axis: int, fits_axis: int
) -> tuple[Optional[fits.BinTableHDU], NDArray]:
    """Processes the time axis, squeezing if its length is 1."""
    if field.shape[np_axis] == 1:
        return None, np.squeeze(field, axis=np_axis)
    else:
        raise NotImplementedError("This functionality is not tested.")
        times = grid.times
        time_centers = (times[:-1] + times[1:]) / 2.0
        header[f"CTYPE{fits_axis}"] = "TIME"
        header[f"CUNIT{fits_axis}"] = times.unit.to_string("fits")
        header[f"CRPIX{fits_axis}"] = 1
        header[f"CRVAL{fits_axis}"] = time_centers[0].value
        header[f"CDELT{fits_axis}"] = (
            (time_centers[1].value - time_centers[0].value)
            if len(time_centers) > 1
            else 0.0
        )
        time_col = fits.Column(
            name="TIME",
            format="E",
            unit=times.unit.to_string("fits"),
            array=time_centers.value,
        )
        return fits.BinTableHDU.from_columns([time_col], name="TIMES"), field


def _process_polarization(
    header: fits.Header, grid: Grid, field: NDArray, np_axis: int, fits_axis: int
) -> tuple[Optional[fits.BinTableHDU], NDArray]:
    """Processes the polarization axis, squeezing if its length is 1."""
    if field.shape[np_axis] == 1:
        return None, np.squeeze(field, axis=np_axis)
    else:
        raise NotImplementedError("This functionality is not tested.")
        stokes_params = grid.polarization.value
        header[f"CTYPE{fits_axis}"] = "STOKES"
        header[f"CRVAL{fits_axis}"] = 1
        header[f"CRPIX{fits_axis}"] = 1
        header[f"CDELT{fits_axis}"] = 1
        max_len = max(len(s) for s in stokes_params)
        pol_col = fits.Column(
            name="POL", format=f"{max_len}A", array=np.array(stokes_params)
        )
        return fits.BinTableHDU.from_columns([pol_col], name="POLARIZATIONS"), field


def _process_sample(
    header: fits.Header, field: NDArray, np_axis: int, fits_axis: int
) -> NDArray:
    """Processes the sample axis, squeezing if its length is 1."""
    if field.shape[np_axis] == 1:
        return np.squeeze(field, axis=np_axis)
    else:
        header[f"CTYPE{fits_axis}"] = "SAMPLE"
        header[f"CRVAL{fits_axis}"] = 1
        header[f"CRPIX{fits_axis}"] = 1
        header[f"CDELT{fits_axis}"] = 1
        return field


# --- Orchestrator and Main Class ---


def _create_spatial_header(grid: Grid) -> fits.Header:
    """Creates a FITS header with spatial WCS, removing the WCSAXES keyword."""
    header = grid.spatial.to_header()
    if "WCSAXES" in header:
        del header["WCSAXES"]
    return header


def _process_dynamic_axes(
    grid: Grid, field: NDArray
) -> tuple[fits.Header, list[fits.BinTableHDU], NDArray]:
    """
    Orchestrates the dynamic processing of non-spatial axes by calling subroutines.
    Processes axes from highest index to lowest to prevent index shifting.
    """
    header = fits.Header()
    extension_hdus = []
    processed_field = field.copy()

    # Fields shape: (sample, pol, time, freq, y, x)
    axes = dict(
        frequency=(3, 3),
        time=(2, 4),
        polarization=(1, 5),
        samples=(0, 6),
    )

    hdu_freq, processed_field = _process_frequency(
        header, grid, processed_field, *axes["frequency"]
    )
    hdu_time, processed_field = _process_time(
        header, grid, processed_field, *axes["time"]
    )
    hdu_pola, processed_field = _process_polarization(
        header, grid, processed_field, *axes["polarization"]
    )
    processed_field = _process_sample(header, processed_field, *axes["samples"])

    return header, extension_hdus, processed_field


# --------------------------------------------------------------------------------------
# Public API ---------------------------------------------------------------------------
# --------------------------------------------------------------------------------------


class FitsSaver:
    """Orchestrates FITS file creation with dynamic axis handling."""

    def __init__(self, grid: Grid, field_samples: NDArray):
        if field_samples.ndim != 6:
            raise ValueError(
                f"Input field must be 6-dimensional, but got {field_samples.ndim} dimensions."
            )
        self.grid = grid
        self.field = field_samples

    def save_mean(self, filename: str, sky_unit: u.Unit | None = None):
        """Averages data and saves, dynamically removing single-entry axes."""
        print(f"\n--- Saving mean to '{filename}' ---")
        # Average over samples, but keep the dimension for consistent processing
        field_to_save = self.field.mean(axis=0, keepdims=True)
        self._save(filename, field_to_save, sky_unit)

    def save_std(self, filename: str, sky_unit: u.Unit | None = None, correct_bias: bool = False):
        """Averages data and saves, dynamically removing single-entry axes."""
        print(f"\n--- Saving mean to '{filename}' ---")
        # Average over samples, but keep the dimension for consistent processing
        field_to_save = self.field.std(axis=0, keepdims=True)
        # Apply Bessel correction if correct_bias is True
        correction = np.sqrt(N/(N-1)) if correct_bias else 1.0
        field_to_save *= correction

        self._save(filename, field_to_save, sky_unit)

    def save_samples(self, filename: str, sky_unit: u.Unit | None = None):
        """Saves sample data, dynamically removing any single-entry axes."""
        print(f"\n--- Saving samples to '{filename}' ---")
        self._save(filename, self.field, sky_unit)

    def _save(self, filename: str, field_data: NDArray, sky_unit: u.Unit | None = None):
        """Generic save method using the dynamic helper functions."""
        spatial_header = _create_spatial_header(self.grid)
        other_header, extensions, final_field = _process_dynamic_axes(
            self.grid, field_data
        )

        final_header = spatial_header
        final_header.update(other_header)
        if sky_unit is not None:
            final_header["BUNIT"] = sky_unit.to_string("fits")

        primary_hdu = fits.PrimaryHDU(data=final_field, header=final_header)
        hdul = fits.HDUList([primary_hdu] + extensions)
        hdul.writeto(filename, overwrite=True)
        print(
            f"Successfully saved FITS file. Final data shape in file: {final_field.shape}"
        )
