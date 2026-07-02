from typing import Optional
import numpy as np
from astropy.io import fits
from astropy import units as u
from jubik.grid import Grid
from numpy.typing import NDArray

__all__ = ["FitsSaver"]

# --------------------------------------------------------------------------------------
# Private Subroutines ------------------------------------------------------------------
# --------------------------------------------------------------------------------------


def _is_uniform(values: NDArray, rtol: float = 1e-4) -> bool:
    """True if `values` are equally spaced (so a linear FITS axis is exact)."""
    if len(values) < 3:
        return True
    steps = np.diff(np.asarray(values, dtype=float))
    return bool(np.allclose(steps, steps[0], rtol=rtol, atol=0.0))


def _freq_tab_hdu(freqs_hz: NDArray, extname: str, colname: str) -> fits.BinTableHDU:
    """Coordinate-array HDU for a FITS -TAB spectral axis (WCS Paper III).

    The lookup table holds the true per-channel frequencies. Per the standard the
    coordinate array has the WCS-axis count (=1) as its last (fastest) dimension and
    the K sample points as the slower one -> numpy cell shape (K, 1), written with
    TDIM '(1,K)' (FITS axis order is reversed w.r.t. numpy).
    """
    k = len(freqs_hz)
    cell = np.asarray(freqs_hz, dtype=np.float64).reshape(1, k, 1)  # (nrows=1, K, 1)
    col = fits.Column(name=colname, format=f"{k}D", dim=f"(1,{k})", array=cell)
    return fits.BinTableHDU.from_columns([col], name=extname)


def _process_frequency(
    header: fits.Header, grid: Grid, field: NDArray, np_axis: int, fits_axis: int
) -> tuple[Optional[fits.BinTableHDU], NDArray]:
    """Processes the frequency axis. This axis is never squeezed.

    Uniform channels get a plain linear axis (exact, read by every tool). Non-uniform
    channels (e.g. two disjoint spectral windows with a gap) cannot be described by a
    single CRVAL/CDELT, so a FITS -TAB lookup-table axis is written instead and the
    accompanying coordinate-array HDU is returned for attachment to the HDUList.
    """
    freqs = u.Quantity(grid.spectral.center).to(u.Hz, equivalencies=u.spectral())

    if field.shape[np_axis] == 1 and np.isinf(freqs[0]):
        return None, np.squeeze(field, axis=np_axis)

    fval = freqs.value
    header[f"CUNIT{fits_axis}"] = "Hz"

    if _is_uniform(fval):
        header[f"CTYPE{fits_axis}"] = "FREQ"
        header[f"CRPIX{fits_axis}"] = 1  # Fits reference 0-th axis is indexed by 1
        header[f"CRVAL{fits_axis}"] = float(fval[0])
        header[f"CDELT{fits_axis}"] = float(fval[1] - fval[0]) if len(fval) > 1 else 0.0
        return None, field

    # Non-uniform -> tabulated (-TAB) axis. CRPIX/CRVAL/CDELT = 1 makes the intermediate
    # coordinate equal the 1-based channel number, which directly indexes the table.
    extname, colname = "WCS-FREQ", "FREQ"
    header[f"CTYPE{fits_axis}"] = "FREQ-TAB"
    header[f"CRPIX{fits_axis}"] = 1
    header[f"CRVAL{fits_axis}"] = 1
    header[f"CDELT{fits_axis}"] = 1
    header[f"PS{fits_axis}_0"] = extname   # coordinate-array extension (EXTNAME)
    header[f"PS{fits_axis}_1"] = colname   # coordinate-array column
    return _freq_tab_hdu(fval, extname, colname), field


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

    # Attach any coordinate-array extensions the axis processors produced (e.g. the
    # -TAB frequency lookup table). Previously these were built and silently dropped.
    extension_hdus = [h for h in (hdu_freq, hdu_time, hdu_pola) if h is not None]

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

    def save_std(self, filename: str, sky_unit: u.Unit | None = None):
        """Averages data and saves, dynamically removing single-entry axes."""
        print(f"\n--- Saving mean to '{filename}' ---")
        # Average over samples, but keep the dimension for consistent processing
        field_to_save = self.field.std(axis=0, keepdims=True)
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
