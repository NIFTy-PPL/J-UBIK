# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2024 Max-Planck-Society

# %

from astropy import units as u

from ...grid import Grid


def data_margin(telescope_config: dict) -> float | str | None:
    """`target.data_arcsec_extension`: "auto" when absent, None for `off` (or false,
    null: the PSF-sized cutout), else the margin in arcsec (>= 0)."""
    raw = telescope_config.get("target", {}).get("data_arcsec_extension", "auto")
    if raw is None or raw is False or raw == "off":
        return None
    if raw == "auto":
        return raw
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw < 0:
        raise ValueError(
            "target.data_arcsec_extension: a number >= 0 (arcsec), 'auto' or 'off', "
            f"got {raw!r}"
        )
    return float(raw)


def get_grid_extension_from_config(
    telescope_config: dict,
    reconstruction_grid: Grid,
) -> tuple[int, int]:
    """Load a pixelwise extension of the reconstruction grid. The data cutout covers
    the reconstruction grid extended by half the grid extension in both spatial
    dimensions.

    `target.data_arcsec_extension` (arcsec, summed over both sides; see
    `data_margin`) sets it. Absent, it is automatic: the extension is 0 here and the
    likelihood builder grows each filter's cutout by the reach of the shift and
    rotation priors (a normal prior reaches 3 sigma; a fit beyond that can still
    move sky light past the edge). `off` gives `psf.psf_arcsec_extension`, the
    PSF-sized cutout of before. `PsfStatic` pads its own FFT, so the cutout needs
    to hold only every data pixel with a subsample point on the sky; data beyond
    that are masked and only cost time.

    Parameters
    ----------
    telescope_config: dict
        The telescope_config dict holds psf_arcsec_extension, which holds the
        extension in units of arcsec, and optionally target.data_arcsec_extension.
    reconstruction_grid: Grid
        The grid underlying the reconstruction.

    Returns
    -------
    grid_extension: tuple[int]
        A pixel number tuple, that specifies by how many pixels the
        reconstruction will be zero padded.
    """

    # TODO : Make this explicit units in the config.
    psf_arcsec_extension = telescope_config["psf"].get("psf_arcsec_extension")
    if psf_arcsec_extension is None:
        raise ValueError("Need to provide either `psf_arcsec_extension`.")
    margin = data_margin(telescope_config)
    extension = (
        psf_arcsec_extension if margin is None else 0.0 if margin == "auto" else margin
    )

    assert len(reconstruction_grid.spatial.pixel_scales_yx) == 2

    return tuple(
        [
            int((extension * u.Unit("arcsec")).to(u.Unit("deg")) / 2 / dist)
            for dist in reconstruction_grid.spatial.pixel_scales_yx
        ]
    )
