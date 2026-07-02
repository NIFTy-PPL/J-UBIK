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
    """Load a pixelwise extension of the reconstruction grid. The reconstruction grid
    will be extended by half the grid extension in both spatial dimensions.

    This extension is needed in order to avoid flux being wrapped around the periodic
    boundary of the grid by the fft-psf convolution.

    Parameters
    ----------
    telescope_config: dict
        The telescope_config dict holds psf_arcsec_extension, which holds the
        extension in units of arcsec.
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

    assert len(reconstruction_grid.spatial.distances) == 2

    return tuple(
        [
            int((psf_arcsec_extension * u.Unit("arcsec")).to(u.Unit("deg")) / 2 / dist)
            for dist in reconstruction_grid.spatial.distances
        ]
    )
