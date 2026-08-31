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


def _process_frequency(
    header: fits.Header, grid: Grid, field: NDArray, np_axis: int, fits_axis: int
) -> tuple[Optional[fits.BinTableHDU], NDArray]:
    """Describe the spectral axis in the header and, if possible, in a table.

    The frequency axis is only squeezed out of the field if it holds a single
    channel whose center is infinite, i.e. if the grid carries no spectral
    information at all.

    The header keywords (``CTYPEn``, ``CRVALn``, ``CDELTn``) can only encode a
    linear frequency axis and are therefore approximate; they are written for
    backwards compatibility. The exact bin centers are additionally returned as
    a ``FREQUENCIES`` binary table.

    Parameters
    ----------
    header : fits.Header
        Header to which the spectral WCS keywords are added, in place.
    grid : Grid
        Grid providing the spectral bin centers via ``grid.spectral.center``.
    field : NDArray
        Field of shape (sample, polarization, time, frequency, y, x).
    np_axis : int
        Index of the frequency axis inside `field`.
    fits_axis : int
        One-based FITS axis number used for the header keyword suffix.

    Returns
    -------
    tuple[Optional[fits.BinTableHDU], NDArray]
        The ``FREQUENCIES`` table (`None` if the axis was squeezed) and the
        possibly squeezed field.
    """
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
    """Describe the time axis in the header, squeezing it if it has length 1.

    Parameters
    ----------
    header : fits.Header
        Header to which the time WCS keywords are added, in place.
    grid : Grid
        Grid providing the time bin bounds via ``grid.times``.
    field : NDArray
        Field of shape (sample, polarization, time, frequency, y, x).
    np_axis : int
        Index of the time axis inside `field`.
    fits_axis : int
        One-based FITS axis number used for the header keyword suffix.

    Returns
    -------
    tuple[Optional[fits.BinTableHDU], NDArray]
        The ``TIMES`` table (`None` if the axis was squeezed) and the possibly
        squeezed field.

    Raises
    ------
    NotImplementedError
        If the time axis is longer than one bin. Multiple time bins are
        untested and hence rejected.
    """
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
    """Describe the polarization axis, squeezing it if it has length 1.

    Parameters
    ----------
    header : fits.Header
        Header to which the Stokes WCS keywords are added, in place.
    grid : Grid
        Grid providing the polarization labels via ``grid.polarization``.
    field : NDArray
        Field of shape (sample, polarization, time, frequency, y, x).
    np_axis : int
        Index of the polarization axis inside `field`.
    fits_axis : int
        One-based FITS axis number used for the header keyword suffix.

    Returns
    -------
    tuple[Optional[fits.BinTableHDU], NDArray]
        The ``POLARIZATIONS`` table (`None` if the axis was squeezed) and the
        possibly squeezed field.

    Raises
    ------
    NotImplementedError
        If more than one polarization is present. This case is untested and
        hence rejected.
    """
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
    """Describe the sample axis, squeezing it if it has length 1.

    The sample axis enumerates posterior samples and carries no physical
    coordinate, hence it is described by a unit-increment index axis.

    Parameters
    ----------
    header : fits.Header
        Header to which the sample WCS keywords are added, in place.
    field : NDArray
        Field of shape (sample, polarization, time, frequency, y, x).
    np_axis : int
        Index of the sample axis inside `field`.
    fits_axis : int
        One-based FITS axis number used for the header keyword suffix.

    Returns
    -------
    NDArray
        The field, with the sample axis squeezed out if it had length 1.
    """
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
    """Create a FITS header holding the spatial WCS of the grid.

    ``WCSAXES`` is removed because the number of axes of the written file is
    only known after the non-spatial axes have been squeezed.

    Parameters
    ----------
    grid : Grid
        Grid providing the spatial world coordinate system.

    Returns
    -------
    fits.Header
        Header with the spatial WCS keywords for axes 1 (x) and 2 (y).
    """
    header = grid.spatial.to_header()
    if "WCSAXES" in header:
        del header["WCSAXES"]
    return header


def _process_dynamic_axes(
    grid: Grid, field: NDArray
) -> tuple[fits.Header, list[fits.BinTableHDU], NDArray]:
    """Describe the non-spatial axes and drop the ones of length one.

    The axes are processed from the highest numpy index to the lowest, such
    that squeezing an axis does not shift the index of an axis not yet
    processed.

    Parameters
    ----------
    grid : Grid
        Grid providing the spectral, temporal and polarization coordinates.
    field : NDArray
        Field of shape (sample, polarization, time, frequency, y, x).

    Returns
    -------
    tuple[fits.Header, list[fits.BinTableHDU], NDArray]
        The header holding the non-spatial WCS keywords, the extension HDUs to
        append to the file, and the field with all length-one axes (except the
        spatial ones) removed.
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
    """Write posterior sky samples, or their statistics, to FITS files.

    The saver takes the full sample stack together with the `Grid` describing
    its coordinates and writes the mean, the standard deviation or all samples.
    Axes of length one are dropped from the written data, so that a single
    Stokes-I, single-time, single-frequency mean ends up as a plain 2D image,
    while the corresponding WCS is taken from the grid.

    Parameters
    ----------
    grid : Grid
        The coordinate system of `field_samples`.
    field_samples : NDArray
        Sky samples of shape (sample, polarization, time, frequency, y, x).

    Raises
    ------
    ValueError
        If `field_samples` is not 6-dimensional.

    Examples
    --------
    >>> saver = FitsSaver(grid, samples)  # samples.shape == (10, 1, 1, 3, 64, 64)
    >>> saver.save_mean("mean.fits", sky_unit=u.Unit("Jy"))
    """

    def __init__(self, grid: Grid, field_samples: NDArray):
        if field_samples.ndim != 6:
            raise ValueError(
                f"Input field must be 6-dimensional, but got {field_samples.ndim} dimensions."
            )
        self.grid = grid
        self.field = field_samples

    def save_mean(self, filename: str, sky_unit: u.Unit | None = None):
        """Save the sample mean, dropping all single-entry axes.

        Parameters
        ----------
        filename : str
            Path of the FITS file to write. An existing file is overwritten.
        sky_unit : u.Unit, optional
            Unit of the sky brightness, written to ``BUNIT``. If `None`, no
            ``BUNIT`` keyword is written.
        """
        print(f"\n--- Saving mean to '{filename}' ---")
        # Average over samples, but keep the dimension for consistent processing
        field_to_save = self.field.mean(axis=0, keepdims=True)
        self._save(filename, field_to_save, sky_unit)

    def save_std(
        self, filename: str, sky_unit: u.Unit | None = None, correct_bias: bool = False
    ):
        """Save the sample standard deviation, dropping all single-entry axes.

        Parameters
        ----------
        filename : str
            Path of the FITS file to write. An existing file is overwritten.
        sky_unit : u.Unit, optional
            Unit of the sky brightness, written to ``BUNIT``. If `None`, no
            ``BUNIT`` keyword is written.
        correct_bias : bool, optional
            If True, apply the Bessel correction ``sqrt(N / (N - 1))`` to turn
            the biased standard deviation into the unbiased one (default:
            False).
        """
        print(f"\n--- Saving mean to '{filename}' ---")
        # Average over samples, but keep the dimension for consistent processing
        field_to_save = self.field.std(axis=0, keepdims=True)
        # Apply Bessel correction if correct_bias is True
        N = self.field.shape[0]
        correction = np.sqrt(N / (N - 1)) if correct_bias else 1.0
        field_to_save *= correction

        self._save(filename, field_to_save, sky_unit)

    def save_samples(self, filename: str, sky_unit: u.Unit | None = None):
        """Save all samples, dropping any single-entry axes.

        Parameters
        ----------
        filename : str
            Path of the FITS file to write. An existing file is overwritten.
        sky_unit : u.Unit, optional
            Unit of the sky brightness, written to ``BUNIT``. If `None`, no
            ``BUNIT`` keyword is written.
        """
        print(f"\n--- Saving samples to '{filename}' ---")
        self._save(filename, self.field, sky_unit)

    def _save(self, filename: str, field_data: NDArray, sky_unit: u.Unit | None = None):
        """Write `field_data` and the grid coordinates to a FITS file.

        Parameters
        ----------
        filename : str
            Path of the FITS file to write. An existing file is overwritten.
        field_data : NDArray
            Field of shape (sample, polarization, time, frequency, y, x).
        sky_unit : u.Unit, optional
            Unit of the sky brightness, written to ``BUNIT``. If `None`, no
            ``BUNIT`` keyword is written.
        """
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
