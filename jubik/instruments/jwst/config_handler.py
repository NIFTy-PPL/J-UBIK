# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2024 Max-Planck-Society

# %

from astropy import units as u

from ...grid import Grid


def get_grid_extension_from_config(
    telescope_config: dict,
    reconstruction_grid: Grid,
) -> tuple[int, int]:
    """Load a pixelwise extension of the reconstruction grid. The data cutout covers
    the reconstruction grid extended by half the grid extension in both spatial
    dimensions.

    The extension is `target.data_arcsec_extension` when given, else
    `psf.psf_arcsec_extension` (both in arcsec, summed over both sides).
    `PsfStatic` pads its own FFT, so the cutout needs to hold only every data pixel
    with a subsample point on the sky: the sky plus about two data pixels and the
    reach of the shift (and rotation) prior, which the build checks when
    `data_arcsec_extension` is given. Data beyond that are masked and only cost time.

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
    extension = telescope_config.get("target", {}).get("data_arcsec_extension")
    if extension is None:
        extension = psf_arcsec_extension

    assert len(reconstruction_grid.spatial.pixel_scales_yx) == 2

    return tuple(
        [
            int((extension * u.Unit("arcsec")).to(u.Unit("deg")) / 2 / dist)
            for dist in reconstruction_grid.spatial.pixel_scales_yx
        ]
    )
